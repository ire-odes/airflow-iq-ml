/*
 * FiltSure Node Firmware -- LoRaWAN + On-Device MFCC + Slot-Synchronised Sampling
 *
 * Based on the LoRaWAN+MFCC sketch. Sensors, audio capture, the MFCC
 * pipeline, payload layout, join/session persistence and deep-sleep
 * structure are unchanged. The only thing reworked is WHEN the node wakes
 * and WHEN it transmits.
 *
 * ===========================================================================
 * WHY: PAIRED NODES MUST SAMPLE THE SAME MOMENT
 *
 * Nodes deploy in pairs -- one blower side, one filter side. The signal is
 * the DIFFERENCE between them, and a difference is only meaningful if both
 * readings describe the same instant. Readings minutes apart, taken during
 * different blower states, aren't comparable at all.
 *
 * The old sleep couldn't give that. goToSleep(SLEEP_SECONDS) sleeps 60s from
 * the moment it's called, so the real period is 60s PLUS however long the
 * node was awake -- and awake time varies with join attempts, the RFID
 * timeout, audio capture and RX windows. Measured on the deployed unit: a
 * rock-steady 74.4s period (stddev 0.1s) from a nominal 60s sleep, i.e.
 * ~14.4s of awake time. Perfectly regular, and perfectly unaligned with any
 * other node, because the phase is fixed by whenever that node happened to
 * boot. Two nodes can never converge. 74.4s doesn't even divide into an hour.
 *
 * THE FIX: sleep to an absolute boundary, not for a duration.
 *
 *     sleep = slot - (unix_now % slot)
 *
 * Both halves of a pair land on the same wall-clock multiples of `slot`
 * regardless of boot time, and because each cycle recomputes from the true
 * clock, awake-time variance is corrected every wake instead of accumulating.
 *
 * Wall-clock time, the slot, and this node's transmit offset all arrive in a
 * config downlink from the ingest_ttn Edge Function. Not from the
 * DeviceTimeReq MAC command: that depends on RadioLib's MAC surface (which
 * moves between versions) and on the network answering it, whereas an
 * application downlink works on any stack that can receive one.
 *
 * ===========================================================================
 * SAMPLING IS SYNCHRONISED; TRANSMISSION DELIBERATELY IS NOT
 *
 * LoRaWAN is ALOHA -- no carrier sense, no collision avoidance. Two
 * co-located nodes transmitting at the same instant collide at the gateway
 * and BOTH packets are lost. Synchronising the radio would destroy the very
 * data this exists to make comparable.
 *
 * So sensors and audio are read immediately on wake, at the boundary and
 * before any radio work (already the order in setup() below), then the
 * uplink is held for a server-assigned offset: blower transmits at +0s,
 * filter at +4s. Assigned from duct_role in the database rather than derived
 * on the node, so re-roling a unit in the UI takes effect without a reflash.
 *
 * Payload budget unchanged: 55 bytes, DR2 (SF8/BW125).
 */

#include <RadioLib.h>
#include <Preferences.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <MFRC522.h>
#include <driver/i2s.h>
#include <esp32-hal-rgb-led.h>
#include <esp_sleep.h>
#include <esp_task_wdt.h>
#include <math.h>
#include <string.h>

/* ===================== FEATURE FLAGS ===================== */
#define USE_BME    1
#define USE_AUDIO  1

/* ===================== PINS (PRESERVED FROM PCB) ===================== */
#define BATTERY_PIN       0
#define DIAG_PIN          2
#define WIND_PIN          3
#define SCK               4
#define MOSI              5
#define REGULATOR_EN_PIN  14
#define RGB_BUILTIN       8
#define MISO              15
#define RFID_RST          19
#define RFID_CS           22
#define BME_CS            23
#define LORA_CS           21
#define LORA_RST          20
#define LORA_DIO0         18
#define LORA_DIO1         12

#define I2S_SCK_PIN   6
#define I2S_WS_PIN    7
#define I2S_SD_PIN    1

/* ===================== TIMING ===================== */
#define us_S              1000000ULL
#define RFID_TIMEOUT_MS   1500

// Used only before the node has ever received a time sync, and for join
// backoff. Unaligned by definition -- the node still reports, and aligns on
// its first config downlink. Degrading to "works but unsynchronised" beats
// not reporting at all.
#define FALLBACK_SLEEP_SECONDS  900

// Compiled-in defaults, overridden by the config downlink. The slot must
// divide evenly into an hour (300/900/1800/3600) or the "boundary" walks
// around the clock and a pair never shares one.
#define DEFAULT_SLOT_SECONDS    900     // 15 min
#define CONFIG_FPORT            10      // must match TTN_CONFIG_FPORT
#define DATA_FPORT              1
#define CONFIG_PAYLOAD_VERSION  0x02
#define MAX_TX_OFFSET_S         30      // sanity bound on the server value

// Hardware watchdog. setup() does everything and never returns -- it ends in
// deep sleep -- so any blocking call inside it hangs the node indefinitely.
// Observed 2026-09-01: both nodes stopped mid-cycle and sat with the red LED
// lit for 21 hours until power-cycled by hand. Red is only ever set
// immediately before a sleep call, so they were awake and stuck, not
// sleeping: something blocked before reaching goToSleep(), most likely a
// RadioLib call waiting on a DIO interrupt that never arrived.
//
// The watchdog turns "dead until someone notices" into "misses one cycle and
// reboots". 180s is comfortably longer than a healthy cycle (RFID 1.5s, audio
// ~1s, join up to ~30s, TX plus RX windows a few more) while still recovering
// well inside a single 900s slot.
#define WDT_TIMEOUT_S           180

/* ===================== TTN CREDENTIALS =====================
 * !!! PER-DEVICE -- CHANGE BEFORE FLASHING THE SECOND NODE !!!
 *
 * devEUI and appKey identify ONE physical node. Flashing this file
 * unmodified to both halves of a pair gives them the same DevEUI, and TTN
 * treats that as a single device: each join invalidates the other's
 * session, so whichever node uplinks next fails its MIC check against keys
 * the network has already replaced. That presents as intermittent "MIC
 * mismatch" on both units, alternating unpredictably -- exactly the
 * symptom that is hardest to attribute, because each node looks fine on
 * its own and fails only when the other happens to join.
 *
 * Do not flash this file directly. firmware/make_lora_nodes.py replaces this
 * whole block with each node's own DevEUI and AppKey (from the gitignored
 * firmware/lora_keys.json) and writes firmware/lora_nodes/<NODE>/. Flash those.
 */
#error "Base sketch: flash a generated sketch from firmware/lora_nodes/ (run firmware/make_lora_nodes.py)."

uint64_t joinEUI = 0x0000000000000000;
uint64_t devEUI  = 0x0000000000000000;
uint8_t  appKey[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
uint8_t  nwkKey[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

/* ===================== RADIO / SENSOR OBJECTS ===================== */
SX1276 radio = new Module(LORA_CS, LORA_DIO0, LORA_RST, LORA_DIO1);
LoRaWANNode node(&radio, &US915, 2);

Adafruit_BME280 bme(BME_CS);
MFRC522         rfid(RFID_CS, RFID_RST);
Preferences     store;

/* ===================== RTC STATE (survives deep sleep) ===================== */
RTC_DATA_ATTR uint16_t bootCount = 0;
RTC_DATA_ATTR uint16_t bootCountSinceUnsuccessfulJoin = 0;
RTC_DATA_ATTR uint8_t  LWsession[RADIOLIB_LORAWAN_SESSION_BUF_SIZE];

// Slot/clock state. RTC RAM survives deep sleep but not power loss, which is
// the correct lifetime: after a power cut the clock is stale anyway and must
// be re-synced before it can be trusted.
RTC_DATA_ATTR uint32_t rtc_slot_seconds   = DEFAULT_SLOT_SECONDS;
RTC_DATA_ATTR uint32_t rtc_tx_offset_s    = 0;
RTC_DATA_ATTR uint32_t rtc_unix_at_sync   = 0;   // wall clock from last downlink
RTC_DATA_ATTR uint32_t rtc_uptime_at_sync = 0;   // millis() when it was taken
RTC_DATA_ATTR bool     rtc_time_valid     = false;

/* ===================== AUDIO / MFCC CONFIG ===================== */
#define I2S_PORT           I2S_NUM_0
#define AUDIO_SAMPLE_RATE  16000
#define CLIP_DURATION_S    1
#define TOTAL_SAMPLES      (AUDIO_SAMPLE_RATE * CLIP_DURATION_S)

#define FRAME_SIZE       400
#define HOP_SIZE         160
#define FFT_SIZE         512
#define NUM_MEL_FILTERS  26
#define NUM_MFCC         13
#define NUM_FRAMES       ((TOTAL_SAMPLES - FRAME_SIZE) / HOP_SIZE + 1)

int16_t audio_samples[TOTAL_SAMPLES];
bool    g_i2sReady = false;

static float melFilterbank[NUM_MEL_FILTERS][FFT_SIZE / 2 + 1];
static float hannWindow[FRAME_SIZE];
static float mfccAccum[NUM_MFCC];
static int16_t mfccOut[NUM_MFCC];

/* ===================== TELEMETRY + MFCC PAYLOAD (unchanged, 55 bytes) ===== */
struct __attribute__((packed)) UplinkPayload {
    uint8_t  mac[6];
    uint16_t boot;
    uint16_t battery_mv;
    int16_t  temp_c100;
    uint16_t humidity100;
    uint32_t pressure_pa;
    uint16_t windspeed100;
    uint8_t  rfid[4];
    uint8_t  filter_ok;
    int16_t  mfcc[NUM_MFCC];
};

/* ===================== SENSOR STATE ===================== */
float  Temp = -1, Humd = -1, Prs = -1;
float  batteryVoltage = 0;
float  windSpeed = 0;
String rfidUID = "";
bool   bmeOK = false;

/* ===================== WIND ===================== */
const float Vref             = 3.3f;
const float dividerGain      = 2.0f;
const int   ADCMax           = 4095;
const int   WIND_OVERSAMPLES = 16;
const float kVPerRPM         = 0.0009f;
const float RPM_TO_MPS       = 0.002094f;
float       ema_rpm          = 0.0f;
const float alpha_ema        = 0.2f;

/* ===================== HELPERS ===================== */
inline void deselectAllSPI() {
    digitalWrite(BME_CS,  HIGH);
    digitalWrite(RFID_CS, HIGH);
    digitalWrite(LORA_CS, HIGH);
}

void enableRegulator() {
    pinMode(REGULATOR_EN_PIN, OUTPUT);
    digitalWrite(REGULATOR_EN_PIN, HIGH);
}

void disableRegulatorForSleep() {
    digitalWrite(REGULATOR_EN_PIN, LOW);
}

void goToSleep(uint32_t seconds) {
    // Carry the wall clock across the sleep.
    //
    // A deep-sleep wake RESETS the CPU: execution re-enters setup() and
    // millis() restarts from zero. rtc_uptime_at_sync, captured during the
    // previous boot, is therefore meaningless afterwards -- and because both
    // are unsigned, (millis() - rtc_uptime_at_sync) on the next boot
    // underflows to roughly 4.29e9, adding ~136 years to the clock. Every
    // subsequent slot calculation is then computed from garbage, which shows
    // up as a node that clearly received its config (its cadence changes) but
    // still wakes at arbitrary times instead of on the boundary.
    //
    // Advancing the stored time by the sleep we are about to take, and
    // rebasing the uptime reference to zero, makes currentUnixTime() on the
    // next boot equal rtc_unix_at_sync + millis()/1000 -- correct, because
    // millis() genuinely does start at zero there. Done inside goToSleep so
    // every sleep path (slot, fallback, join backoff) is covered.
    if (rtc_time_valid) {
        rtc_unix_at_sync   = currentUnixTime() + seconds;   // uses the OLD base
        rtc_uptime_at_sync = 0;
    }
    Serial0.printf("[INFO] Sleeping for %lu s...\n", (unsigned long)seconds);
    // Detach before sleeping. Deep sleep powers the CPU down so the watchdog
    // cannot fire during it either way, but unsubscribing keeps the intent
    // explicit: every armed period is one we actually expect to be running.
    esp_task_wdt_delete(NULL);
    store.end();
    disableRegulatorForSleep();
    esp_sleep_enable_timer_wakeup((uint64_t)seconds * us_S);
    esp_deep_sleep_start();
}

/* ===================== TIME + SLOT ===================== */
// Best estimate of wall clock: last synced value plus elapsed uptime.
// millis() is not a disciplined clock, which is why the server resyncs
// periodically -- but ESP32 drift is seconds per day against a slot measured
// in minutes, so this stays accurate between syncs.
uint32_t currentUnixTime() {
    if (!rtc_time_valid) return 0;
    return rtc_unix_at_sync + ((millis() - rtc_uptime_at_sync) / 1000UL);
}

void sleepUntilNextSlot() {
    uint32_t now  = currentUnixTime();
    uint32_t slot = rtc_slot_seconds;

    if (now == 0 || slot == 0) {
        Serial0.println(F("[SLOT] No time sync yet -- unaligned fallback sleep"));
        goToSleep(FALLBACK_SLEEP_SECONDS);
        return;
    }

    uint32_t sleep_s = slot - (now % slot);
    // Waking within a second of the boundary would let the next cycle fire
    // twice inside the same slot; skip to the following one.
    if (sleep_s < 2) sleep_s += slot;

    Serial0.printf("[SLOT] now=%lu slot=%lu -> wake at %lu (in %lu s)\n",
                   (unsigned long)now, (unsigned long)slot,
                   (unsigned long)(now + sleep_s), (unsigned long)sleep_s);
    goToSleep(sleep_s);
}

/* ===================== CONFIG DOWNLINK =====================
 * 10 bytes from queueSlotDownlink() in supabase/functions/ingest_ttn:
 *   [0]    version (0x02)
 *   [1..4] slot seconds,  big-endian
 *   [5..8] unix time,     big-endian
 *   [9]    transmit offset seconds (0 = blower, 4 = filter)
 */
void handleConfigDownlink(uint8_t* data, size_t len) {
    if (len < 10 || data[0] != CONFIG_PAYLOAD_VERSION) {
        Serial0.printf("[CFG] Ignoring downlink (len=%u, ver=0x%02X)\n",
                       (unsigned)len, len ? data[0] : 0);
        return;
    }

    uint32_t slot = ((uint32_t)data[1] << 24) | ((uint32_t)data[2] << 16) |
                    ((uint32_t)data[3] << 8)  |  (uint32_t)data[4];
    uint32_t unix = ((uint32_t)data[5] << 24) | ((uint32_t)data[6] << 16) |
                    ((uint32_t)data[7] << 8)  |  (uint32_t)data[8];
    uint32_t off  = data[9];

    // Bound everything: a corrupt slot of 0 would busy-loop the node awake,
    // and a wildly wrong clock would push the next wake years out. 1.6e9 is
    // 2020-09, so anything earlier is certainly not a real timestamp.
    // Upper bound tightened from 86400 (24h). A corrupted downlink that still
    // passed the version and length checks could previously set a 24-hour
    // slot, which from the outside is indistinguishable from a dead node.
    // Nothing legitimate asks for more than an hour.
    if (slot >= 60 && slot <= 3600) {
        if (slot != rtc_slot_seconds) {
            Serial0.printf("[CFG] Slot %lu -> %lu s\n",
                           (unsigned long)rtc_slot_seconds, (unsigned long)slot);
        }
        rtc_slot_seconds = slot;
    }
    if (off <= MAX_TX_OFFSET_S) {
        rtc_tx_offset_s = off;
    }
    if (unix > 1600000000UL) {
        rtc_unix_at_sync   = unix;
        rtc_uptime_at_sync = millis();
        rtc_time_valid     = true;
        Serial0.printf("[CFG] Time synced: unix=%lu, tx offset=%lu s\n",
                       (unsigned long)unix, (unsigned long)rtc_tx_offset_s);
    }
}

/* ===================== RFID ===================== */
bool waitForRFID(String &uidHex, uint32_t timeout_ms) {
    uidHex = "";
    unsigned long t0 = millis();
    while (millis() - t0 < timeout_ms) {
        if (rfid.PICC_IsNewCardPresent() && rfid.PICC_ReadCardSerial()) {
            for (byte i = 0; i < rfid.uid.size; i++) {
                if (rfid.uid.uidByte[i] < 0x10) uidHex += "0";
                uidHex += String(rfid.uid.uidByte[i], HEX);
            }
            uidHex.toUpperCase();
            rfid.PICC_HaltA();
            rfid.PCD_StopCrypto1();
            return true;
        }
        delay(50);
    }
    return false;
}

/* ===================== WIND ===================== */
float readWindSpeed_mps() {
    uint32_t acc = 0;
    for (int i = 0; i < WIND_OVERSAMPLES; ++i) {
        acc += analogRead(WIND_PIN);
        delayMicroseconds(150);
    }
    float adc   = acc / float(WIND_OVERSAMPLES);
    float volts = (adc / ADCMax) * Vref * dividerGain;
    float rpm   = (kVPerRPM > 0.0f) ? (volts / kVPerRPM) : 0.0f;
    ema_rpm     = alpha_ema * rpm + (1.0f - alpha_ema) * ema_rpm;
    return RPM_TO_MPS * ema_rpm;
}

/* ===================== SENSORS ===================== */
void readSensors() {
    deselectAllSPI();

    rfid.PCD_Init();
    rfid.PCD_AntennaOn();
    rfid.PCD_SetAntennaGain(rfid.RxGain_max);
    waitForRFID(rfidUID, RFID_TIMEOUT_MS);
    deselectAllSPI();

#if USE_BME
    digitalWrite(BME_CS, LOW);
    delay(5);
    if (bme.begin()) {
        Temp  = bme.readTemperature();
        Humd  = bme.readHumidity();
        Prs   = bme.readPressure() / 100.0f;
        bmeOK = true;
    } else {
        Serial0.println(F("[WARN] BME280 init failed"));
        bmeOK = false;
    }
    digitalWrite(BME_CS, HIGH);
#endif

    windSpeed = readWindSpeed_mps();
}

/* ===================== I2S AUDIO CAPTURE ===================== */
void i2s_audio_init() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = AUDIO_SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 256,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0
    };

    i2s_pin_config_t pins = {
        .mck_io_num   = I2S_PIN_NO_CHANGE,
        .bck_io_num   = I2S_SCK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD_PIN
    };

    esp_err_t err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
    if (err != ESP_OK) {
        Serial0.printf("[WARN] I2S install failed: %s\n", esp_err_to_name(err));
        g_i2sReady = false;
        return;
    }
    err = i2s_set_pin(I2S_PORT, &pins);
    if (err != ESP_OK) {
        Serial0.printf("[WARN] I2S set_pin failed: %s\n", esp_err_to_name(err));
        g_i2sReady = false;
        return;
    }
    i2s_zero_dma_buffer(I2S_PORT);
    g_i2sReady = true;
    Serial0.println(F("[AUDIO] I2S mic init done"));
}

void captureAudioClip() {
    int32_t raw_chunk[256];
    size_t  samples_written = 0;

    Serial0.println(F("[AUDIO] Recording 1s clip..."));
    unsigned long t0 = millis();

    while (samples_written < TOTAL_SAMPLES) {
        if (millis() - t0 > 10000) {
            Serial0.println(F("[WARN] Audio capture timed out"));
            break;
        }
        size_t bytes_read = 0;
        esp_err_t rc = i2s_read(I2S_PORT, raw_chunk, sizeof(raw_chunk), &bytes_read, pdMS_TO_TICKS(1000));
        if (rc != ESP_OK) {
            Serial0.printf("[WARN] i2s_read error: %s\n", esp_err_to_name(rc));
            break;
        }
        size_t samples_read = bytes_read / sizeof(int32_t);
        for (size_t i = 0; i < samples_read && samples_written < TOTAL_SAMPLES; i++) {
            audio_samples[samples_written++] = (int16_t)(raw_chunk[i] >> 16);
        }
    }
    Serial0.printf("[AUDIO] Capture done: %u/%u samples\n",
                   (unsigned)samples_written, (unsigned)TOTAL_SAMPLES);
}

/* ===================== MINIMAL FFT ===================== */
static void fftSwap(float &a, float &b) { float t = a; a = b; b = t; }

void fftRadix2(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { fftSwap(re[i], re[j]); fftSwap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cwr = 1.0f, cwi = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float ur = re[i + k],           ui = im[i + k];
                float vr = re[i + k + len/2] * cwr - im[i + k + len/2] * cwi;
                float vi = re[i + k + len/2] * cwi + im[i + k + len/2] * cwr;
                re[i + k]           = ur + vr; im[i + k]           = ui + vi;
                re[i + k + len/2]   = ur - vr; im[i + k + len/2]   = ui - vi;
                float nwr = cwr * wr - cwi * wi;
                float nwi = cwr * wi + cwi * wr;
                cwr = nwr; cwi = nwi;
            }
        }
    }
}

/* ===================== MEL FILTERBANK + WINDOW ===================== */
static float hzToMel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float melToHz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

void buildMelFilterbank() {
    float lowMel  = hzToMel(0.0f);
    float highMel = hzToMel(AUDIO_SAMPLE_RATE / 2.0f);

    float melPoints[NUM_MEL_FILTERS + 2];
    int   binPoints[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        melPoints[i] = lowMel + (highMel - lowMel) * i / (NUM_MEL_FILTERS + 1);
        binPoints[i] = (int)floorf((FFT_SIZE + 1) * melToHz(melPoints[i]) / AUDIO_SAMPLE_RATE);
    }

    memset(melFilterbank, 0, sizeof(melFilterbank));
    for (int m = 1; m <= NUM_MEL_FILTERS; m++) {
        int fMinus = binPoints[m - 1];
        int fCtr   = binPoints[m];
        int fPlus  = binPoints[m + 1];

        for (int k = fMinus; k < fCtr; k++)
            if (k >= 0 && k <= FFT_SIZE / 2)
                melFilterbank[m - 1][k] = (float)(k - fMinus) / (float)(fCtr - fMinus + 1e-6f);

        for (int k = fCtr; k < fPlus; k++)
            if (k >= 0 && k <= FFT_SIZE / 2)
                melFilterbank[m - 1][k] = (float)(fPlus - k) / (float)(fPlus - fCtr + 1e-6f);
    }
}

void buildHannWindow() {
    for (int i = 0; i < FRAME_SIZE; i++)
        hannWindow[i] = 0.5f - 0.5f * cosf(2.0f * PI * i / (FRAME_SIZE - 1));
}

/* ===================== MFCC ===================== */
void computeMFCC() {
    static float fftRe[FFT_SIZE], fftIm[FFT_SIZE];
    static float melEnergy[NUM_MEL_FILTERS];

    memset(mfccAccum, 0, sizeof(mfccAccum));
    int framesUsed = 0;

    for (int f = 0; f < NUM_FRAMES; f++) {
        int start = f * HOP_SIZE;
        if (start + FRAME_SIZE > TOTAL_SAMPLES) break;

        memset(fftRe, 0, sizeof(fftRe));
        memset(fftIm, 0, sizeof(fftIm));
        for (int i = 0; i < FRAME_SIZE; i++) {
            fftRe[i] = (audio_samples[start + i] / 32768.0f) * hannWindow[i];
        }

        fftRadix2(fftRe, fftIm, FFT_SIZE);

        for (int m = 0; m < NUM_MEL_FILTERS; m++) {
            float sum = 0.0f;
            for (int k = 0; k <= FFT_SIZE / 2; k++) {
                float power = fftRe[k] * fftRe[k] + fftIm[k] * fftIm[k];
                sum += power * melFilterbank[m][k];
            }
            melEnergy[m] = log10f(sum + 1e-6f);
        }

        for (int c = 0; c < NUM_MFCC; c++) {
            float sum = 0.0f;
            for (int m = 0; m < NUM_MEL_FILTERS; m++) {
                sum += melEnergy[m] * cosf((PI / NUM_MEL_FILTERS) * (m + 0.5f) * c);
            }
            mfccAccum[c] += sum;
        }
        framesUsed++;
    }

    Serial0.println(F("[MFCC] Coefficients (mean-pooled):"));
    for (int c = 0; c < NUM_MFCC; c++) {
        float avg = mfccAccum[c] / (float)max(1, framesUsed);
        float scaled = avg * 100.0f;
        if (scaled > 32767.0f)  scaled = 32767.0f;
        if (scaled < -32768.0f) scaled = -32768.0f;
        mfccOut[c] = (int16_t)scaled;
        Serial0.printf("  mfcc[%2d] = %.3f (x100 -> %d)\n", c, avg, mfccOut[c]);
    }
}

/* ===================== PAYLOAD BUILDER ===================== */
UplinkPayload buildPayload() {
    UplinkPayload p = {};

    uint64_t mac64 = ESP.getEfuseMac();
    p.mac[0] = (mac64 >>  0) & 0xFF;
    p.mac[1] = (mac64 >>  8) & 0xFF;
    p.mac[2] = (mac64 >> 16) & 0xFF;
    p.mac[3] = (mac64 >> 24) & 0xFF;
    p.mac[4] = (mac64 >> 32) & 0xFF;
    p.mac[5] = (mac64 >> 40) & 0xFF;

    p.boot         = bootCount;
    p.battery_mv   = (uint16_t)(batteryVoltage * 1000.0f);
    p.temp_c100    = bmeOK ? (int16_t)(Temp * 100.0f)  : 0;
    p.humidity100  = bmeOK ? (uint16_t)(Humd * 100.0f) : 0;
    p.pressure_pa  = bmeOK ? (uint32_t)(Prs  * 100.0f) : 0;
    p.windspeed100 = (uint16_t)(windSpeed * 100.0f);

    if (rfidUID.length() >= 8) {
        for (int i = 0; i < 4; i++) {
            String byteStr = rfidUID.substring(i * 2, i * 2 + 2);
            p.rfid[i] = (uint8_t)strtol(byteStr.c_str(), nullptr, 16);
        }
    }
    p.filter_ok = bmeOK ? 1 : 0;

    memcpy(p.mfcc, mfccOut, sizeof(mfccOut));

    Serial0.println(F("\n[PAYLOAD]"));
    Serial0.printf("  MAC       : %02X:%02X:%02X:%02X:%02X:%02X\n",
                   p.mac[0], p.mac[1], p.mac[2], p.mac[3], p.mac[4], p.mac[5]);
    Serial0.printf("  Boot      : %u\n", p.boot);
    Serial0.printf("  Battery   : %u mV\n", p.battery_mv);
    Serial0.printf("  Temp      : %.2f C\n", p.temp_c100 / 100.0f);
    Serial0.printf("  Humidity  : %.2f %%\n", p.humidity100 / 100.0f);
    Serial0.printf("  Pressure  : %lu Pa\n", (unsigned long)p.pressure_pa);
    Serial0.printf("  Wind      : %.2f m/s\n", p.windspeed100 / 100.0f);
    Serial0.printf("  RFID      : %02X%02X%02X%02X\n", p.rfid[0], p.rfid[1], p.rfid[2], p.rfid[3]);
    Serial0.printf("  FilterOK  : %u\n", p.filter_ok);
    Serial0.printf("  Total size: %u bytes\n", (unsigned)sizeof(p));

    return p;
}

/* ===================== LORAWAN CONNECT (unchanged) ===================== */
int16_t lorawanConnect() {
    int16_t state = RADIOLIB_ERR_UNKNOWN;

    node.beginOTAA(joinEUI, devEUI, nwkKey, appKey);

    store.begin("radiolib", false);
    Serial0.println(F("[LORA] Recalling nonces & session..."));

    if (store.isKey("nonces")) {
        uint8_t noncesBuf[RADIOLIB_LORAWAN_NONCES_BUF_SIZE];
        store.getBytes("nonces", noncesBuf, RADIOLIB_LORAWAN_NONCES_BUF_SIZE);
        state = node.setBufferNonces(noncesBuf);
        if (state != RADIOLIB_ERR_NONE) {
            Serial0.printf("[WARN] Nonces restore failed: %d\n", state);
        }

        state = node.setBufferSession(LWsession);
        if (state == RADIOLIB_ERR_NONE && bootCount > 1) {
            Serial0.println(F("[LORA] Session restored -- activating without rejoin"));
            state = node.activateOTAA();
            if (state == RADIOLIB_LORAWAN_SESSION_RESTORED) {
                Serial0.println(F("[LORA] Session active!"));
                store.end();
                bootCountSinceUnsuccessfulJoin = 0;
                return state;
            }
            Serial0.printf("[WARN] Session restore failed (%d) -- will rejoin\n", state);
        }
    } else {
        Serial0.println(F("[LORA] No nonces saved -- fresh device"));
    }

    Serial0.println(F("[LORA] Performing OTAA join..."));
    neopixelWrite(RGB_BUILTIN, 255, 165, 0);

    state = RADIOLIB_ERR_NETWORK_NOT_JOINED;
    while (state != RADIOLIB_LORAWAN_NEW_SESSION) {
        state = node.activateOTAA();

        if (state == RADIOLIB_LORAWAN_NEW_SESSION) {
            Serial0.println(F("[LORA] Joined!"));
            uint8_t noncesBuf[RADIOLIB_LORAWAN_NONCES_BUF_SIZE];
            memcpy(noncesBuf, node.getBufferNonces(), RADIOLIB_LORAWAN_NONCES_BUF_SIZE);
            store.putBytes("nonces", noncesBuf, RADIOLIB_LORAWAN_NONCES_BUF_SIZE);
            Serial0.println(F("[LORA] Nonces saved to flash"));
            bootCountSinceUnsuccessfulJoin = 0;
            break;
        }

        uint32_t backoff = min((bootCountSinceUnsuccessfulJoin++ + 1UL) * 60UL, 3UL * 60UL);
        Serial0.printf("[ERR] Join failed (%d). Retry in %lu s\n", state, (unsigned long)backoff);
        neopixelWrite(RGB_BUILTIN, 255, 0, 0);
        goToSleep(backoff);   // does not return
    }

    store.end();
    return state;
}

/* ===================== SETUP ===================== */
void setup() {
    neopixelWrite(RGB_BUILTIN, 0, 0, 255);
    Serial0.begin(115200);
    delay(500);

    // Armed before any radio or sensor work, so a hang anywhere in setup()
    // triggers a reset rather than parking the node. panic=true makes the
    // timeout a genuine reboot; RTC state (clock, slot, session) survives it,
    // so the node resumes rather than restarting calibration.
    // ESP-IDF v5 takes a config struct here; the older
    // esp_task_wdt_init(timeout, panic) two-argument form was removed and
    // will not compile against idf-release_v5.x toolchains.
    //
    // The Arduino core may already have started the task watchdog, in which
    // case init returns ESP_ERR_INVALID_STATE and the settings must be
    // applied with reconfigure instead -- otherwise the timeout below is
    // silently ignored and the node keeps whatever the core set.
    esp_task_wdt_config_t wdt_cfg = {
        .timeout_ms     = WDT_TIMEOUT_S * 1000,
        .idle_core_mask = 0,      // watch this task only, not the idle tasks
        .trigger_panic  = true,   // reset on timeout rather than just warn
    };
    if (esp_task_wdt_init(&wdt_cfg) == ESP_ERR_INVALID_STATE) {
        esp_task_wdt_reconfigure(&wdt_cfg);
    }
    esp_task_wdt_add(NULL);

    enableRegulator();
    ++bootCount;
    Serial0.printf("\n===== FiltSure LoRaWAN+MFCC Node | Boot #%u =====\n", bootCount);
    if (rtc_time_valid) {
        Serial0.printf("[SLOT] slot=%lu s, tx offset=%lu s, clock=%lu\n",
                       (unsigned long)rtc_slot_seconds, (unsigned long)rtc_tx_offset_s,
                       (unsigned long)currentUnixTime());
    } else {
        Serial0.println(F("[SLOT] No time sync yet -- awaiting config downlink"));
    }

    SPI.begin(SCK, MISO, MOSI);
    pinMode(BME_CS,  OUTPUT); digitalWrite(BME_CS,  HIGH);
    pinMode(RFID_CS, OUTPUT); digitalWrite(RFID_CS, HIGH);
    pinMode(LORA_CS, OUTPUT); digitalWrite(LORA_CS, HIGH);

    batteryVoltage = analogRead(BATTERY_PIN) * (Vref / ADCMax) * dividerGain;
    Serial0.printf("[BATT] %.3f V\n", batteryVoltage);

    // Sensors and audio are read FIRST -- at the slot boundary, before any
    // radio work. This is what makes a pair's readings comparable: everything
    // after this point (join, transmit offset, RX windows) varies in duration
    // and would smear the sample time if it came first.
    readSensors();
    esp_task_wdt_reset();
    Serial0.printf("[SENSOR] Temp=%.2f Hum=%.2f Prs=%.1f Wind=%.2f RFID=%s\n",
                   Temp, Humd, Prs, windSpeed, rfidUID.c_str());

#if USE_AUDIO
    buildHannWindow();
    buildMelFilterbank();
    i2s_audio_init();
    if (g_i2sReady) {
        captureAudioClip();
        esp_task_wdt_reset();
        computeMFCC();
        esp_task_wdt_reset();
    } else {
        Serial0.println(F("[AUDIO] Skipping capture -- I2S init failed, MFCCs will be zero"));
        memset(mfccOut, 0, sizeof(mfccOut));
    }
#else
    memset(mfccOut, 0, sizeof(mfccOut));
#endif

    deselectAllSPI();
    int16_t state = radio.begin();
    if (state != RADIOLIB_ERR_NONE) {
        Serial0.printf("[ERR] Radio init failed: %d\n", state);
        neopixelWrite(RGB_BUILTIN, 255, 0, 0);
        sleepUntilNextSlot();
        return;
    }
    radio.setSyncWord(0x34);

    esp_task_wdt_reset();
    state = lorawanConnect();
    esp_task_wdt_reset();
    if (state != RADIOLIB_LORAWAN_NEW_SESSION &&
        state != RADIOLIB_LORAWAN_SESSION_RESTORED) {
        Serial0.printf("[ERR] Could not activate: %d\n", state);
        neopixelWrite(RGB_BUILTIN, 255, 0, 0);
        sleepUntilNextSlot();
        return;
    }

    node.setDutyCycle(false);
    node.setDwellTime(false);
    node.setADR(false);
    node.setDatarate(2);   // DR2 = SF8/BW125 -- needed for the 55-byte payload

    if (state == RADIOLIB_LORAWAN_NEW_SESSION) {
        Serial0.println(F("[LORA] Waiting 5 s post-join..."));
        delay(5000);
    }

    UplinkPayload payload = buildPayload();

    // Hold the uplink for this node's assigned slice. Blower transmits at +0,
    // filter at +4 -- the sample is already taken, so this delays only the
    // radio and keeps the pair from colliding at the gateway. Skipped until
    // the clock is synced: an unaligned node isn't sharing a boundary with
    // anything yet, so there's nothing to collide with and no reason to burn
    // the extra awake time.
    if (rtc_time_valid && rtc_tx_offset_s > 0) {
        Serial0.printf("[TX] Holding %lu s for transmit slot\n", (unsigned long)rtc_tx_offset_s);
        delay(rtc_tx_offset_s * 1000UL);
    }

    Serial0.printf("[TX] Sending %u bytes (DR2)...\n", (unsigned)sizeof(payload));

    // NOTE: verify this overload against your RadioLib build. On 6.6.0 it is
    // sendReceive(dataUp, lenUp, fPort, dataDown, lenDown, isConfirmed).
    // Downlink presence is judged by downLen rather than the return value,
    // because the return-code convention for "uplink sent, nothing received"
    // has changed across RadioLib releases -- checking the length works either
    // way.
    // downLen MUST start at 0. RadioLib treats it as a pure output -- it
    // writes the received length there and leaves it alone when nothing
    // arrives. Seeding it with sizeof(downBuf) as a "capacity" (the obvious
    // reading of the API, and what this originally did) means a cycle with no
    // downlink reports 64 bytes of uninitialised stack: observed on P4 as
    // "[CFG] Ignoring downlink (len=64, ver=0x78)" -- 0x78 being ASCII 'x',
    // i.e. leftover garbage rather than any real payload.
    uint8_t downBuf[64];
    size_t  downLen = 0;
    state = node.sendReceive((uint8_t*)&payload, sizeof(payload),
                             DATA_FPORT, downBuf, &downLen);

    if (state >= RADIOLIB_ERR_NONE) {
        memcpy(LWsession, node.getBufferSession(), RADIOLIB_LORAWAN_SESSION_BUF_SIZE);
        Serial0.println(F("[OK] Uplink sent -- session saved to RTC"));
        neopixelWrite(RGB_BUILTIN, 0, 255, 0);

        // Clamped as well as zero-initialised: a length longer than the
        // buffer could only come from a library contract change or a
        // corrupted read, and reading past downBuf would be far worse than
        // dropping the packet.
        if (downLen > 0 && downLen <= sizeof(downBuf)) {
            Serial0.printf("[RX] Downlink %u bytes\n", (unsigned)downLen);
            handleConfigDownlink(downBuf, downLen);
        } else if (downLen > sizeof(downBuf)) {
            Serial0.printf("[RX] Ignoring implausible downlink length %u\n", (unsigned)downLen);
        }
    } else {
        Serial0.printf("[ERR] Uplink failed: %d\n", state);
        neopixelWrite(RGB_BUILTIN, 255, 0, 0);
    }

    delay(2000);
    sleepUntilNextSlot();
}

void loop() {}
