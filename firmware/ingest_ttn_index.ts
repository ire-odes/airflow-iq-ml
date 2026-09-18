/**
 * ttn_webhook — Supabase Edge Function
 *
 * Receives uplink webhooks from The Things Network, looks up the device
 * by MAC address in the devices table, then inserts a row into sensor_logs.
 *
 * TTN Console setup:
 *   Integrations → Webhooks → Custom webhook
 *   Base URL  : https://<project>.functions.supabase.co/functions/v1/ingest_ttn
 *   Headers   : Authorization: Bearer <supabase-anon-key>
 *   Messages  : ✅ Uplink message
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

/**
 * Pushes the device's configured sample slot back as a LoRaWAN downlink.
 *
 * LoRaWAN devices can't fetch config the way WiFi devices poll
 * devices.effective_wake_seconds over REST -- they have no IP -- so
 * settings have to ride a downlink. Downlinks can only be *queued*; the
 * network delivers one in the RX window that follows an uplink, which is
 * why this runs here rather than on a timer.
 *
 * "replace" rather than "push" deliberately: the queue should hold at most
 * the current desired slot. Pushing would let stale values pile up behind
 * a device that was offline and deliver a whole backlog when it returns.
 *
 * Best-effort. The sensor data is already written by the time this runs
 * and matters more than the config echo, so every failure path here logs
 * and returns instead of throwing. Silently disabled when the TTN env vars
 * aren't set, so existing deployments keep working untouched.
 */
async function queueSlotDownlink(devEui: string, slotSeconds: number, txOffsetSeconds: number): Promise<string | null> {
  const apiKey = Deno.env.get("TTN_API_KEY");
  const appId = Deno.env.get("TTN_APP_ID");
  const region = Deno.env.get("TTN_REGION") ?? "nam1";
  const fPort = Number(Deno.env.get("TTN_CONFIG_FPORT") ?? "10");

  if (!apiKey || !appId) return "TTN_API_KEY/TTN_APP_ID not set";

  // 10 bytes: version, slot seconds, current Unix time, transmit offset.
  //
  // The time is what lets the node align to an absolute boundary. It could
  // come from the LoRaWAN DeviceTimeReq MAC command instead, but that
  // depends on RadioLib's MAC-command surface (which moves between
  // versions) and on the network server answering it. Sending it at the
  // application layer works on any stack that can receive a downlink.
  //
  // It arrives one RX window late -- roughly 1-5s after the uplink -- so
  // the node's clock is stale by about that much. Irrelevant at slot sizes
  // of minutes: every node is off by a similar small amount, and they
  // still land on the same boundary within a few seconds of each other.
  // The transmit offset is what keeps a blower/filter pair from colliding:
  // both wake and sample on the same boundary, then transmit this many
  // seconds apart. Server-assigned from duct_role rather than derived on the
  // node, so re-roling a unit in the UI takes effect without a reflash.
  const unixNow = Math.floor(Date.now() / 1000);
  const payload = new Uint8Array([
    0x02, // payload version, so the node can reject formats it can't parse
    (slotSeconds >>> 24) & 0xff,
    (slotSeconds >>> 16) & 0xff,
    (slotSeconds >>> 8) & 0xff,
    slotSeconds & 0xff,
    (unixNow >>> 24) & 0xff,
    (unixNow >>> 16) & 0xff,
    (unixNow >>> 8) & 0xff,
    unixNow & 0xff,
    txOffsetSeconds & 0xff,
  ]);
  const frmPayload = btoa(String.fromCharCode(...payload));

  const url = `https://${region}.cloud.thethings.network/api/v3/as/applications/` +
    `${appId}/devices/${devEui}/down/replace`;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        downlinks: [{
          f_port: fPort,
          frm_payload: frmPayload,
          priority: "LOW", // config, not an alarm -- never displace real traffic
        }],
      }),
    });
    if (!res.ok) {
      const body = (await res.text()).slice(0, 300);
      console.error("TTN downlink queue failed:", res.status, body);
      return `HTTP ${res.status}: ${body}`;
    }
    return null;   // queued successfully
  } catch (e) {
    console.error("TTN downlink queue error:", e);
    return `fetch failed: ${e instanceof Error ? e.message : String(e)}`;
  }
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  const uplink = body?.uplink_message as Record<string, unknown> | undefined;
  if (!uplink) {
    return new Response("Missing uplink_message field", { status: 400 });
  }

  const decoded = uplink?.decoded_payload as Record<string, unknown> | undefined;
  if (!decoded || Object.keys(decoded).length === 0) {
    return new Response("decoded_payload is empty — check your TTN payload formatter", { status: 400 });
  }

  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  );

  // Look up device_id from devices table using the MAC address in the payload
  const mac = (decoded.mac as string | undefined)?.replaceAll(":", "");
  if (!mac) {
    return new Response("decoded_payload missing mac field", { status: 400 });
  }

  const { data: device, error: deviceError } = await supabase
    .from("devices")
    .select("id, sample_slot_seconds, sample_slot_synced_at, lora_slot_seconds, lora_tx_offset_seconds")
    .eq("device_mac", mac)
    .single();

  if (deviceError || !device) {
    console.error("Device lookup error:", deviceError);
    return new Response(`Device not found for MAC ${mac}`, { status: 404 });
  }

  const row = {
    device_id:     device.id,
    boot:          decoded.boot                                ?? null,
    battery:       decoded.battery_v                          ?? null,
    temp_c:        decoded.temp_c                             ?? null,
    humidity:      decoded.humidity                           ?? null,
    pressure_pa:   decoded.pressure_pa                        ?? null,
    windSpeed:     decoded.windspeed_mps                      ?? null,
    rfid:          decoded.rfid                               ?? null,
    filter_status: decoded.filter_ok === true ? "Success" : "Fail",
    recorded_at:   uplink?.received_at ?? new Date().toISOString(),
  };

  const { error: insertError } = await supabase
    .from("sensor_logs")
    .insert(row);

  if (insertError) {
    console.error("Supabase insert error:", insertError);
    return new Response(`DB insert failed: ${insertError.message}`, { status: 500 });
  }

  // Bandwidth-limited LoRaWAN devices send MFCC (mel-frequency cepstral
  // coefficient) features instead of a raw audio file — see the formatter's
  // `mfcc` field, 13 floats. One row per device (its primary key is the MAC),
  // overwritten on every uplink; audio_logs.device_id stores the MAC as
  // text, not the devices.id uuid. Best-effort: a bad/missing mfcc field
  // shouldn't fail the sensor-data write above, which already succeeded.
  const mfcc = decoded.mfcc as number[] | undefined;
  if (Array.isArray(mfcc) && mfcc.length > 0) {
    const { error: audioError } = await supabase
      .from("audio_logs")
      .upsert(
        { device_id: mac, is_lora: true, mfcc_coefficients: mfcc, storage_path: null, updated_at: new Date().toISOString() },
        { onConflict: "device_id" },
      );
    if (audioError) console.error("audio_logs upsert error:", audioError);
  }

  // Queue the sampling slot for this device. The device id TTN addresses is
  // the end-device id from the uplink envelope, which is not necessarily the
  // MAC in the payload, so take it from end_device_ids rather than reusing
  // `mac`. Skipped entirely when no slot is configured, leaving the device
  // on its compiled-in default.
  //
  // Rationed rather than sent every uplink: TTN's fair use policy allows
  // roughly 10 downlinks per device per day, and at a 15-minute slot a
  // node uplinks ~96 times daily. Resyncing every RESYNC_HOURS keeps this
  // to ~2/day. That's ample -- the ESP32 RTC drifts on the order of
  // seconds per day, which is nothing against a slot measured in minutes.
  const RESYNC_HOURS = 12;

  // A node that rejoins gets a NEW session, and the network drops any
  // downlink still queued under the old session keys -- so a config downlink
  // queued just before a rejoin is silently lost. Since the timestamp below
  // is stamped on queueing rather than on delivery, that node would then wait
  // the full RESYNC_HOURS with no slot and no clock, sleeping unaligned the
  // whole time. Observed on P9: queued 12:08, node power-cycled to Boot #1,
  // rejoined, and reported "No time sync yet".
  //
  // The boot counter is the tell. It lives in RTC memory, which survives deep
  // sleep but not power loss, so a value at or below the last one we saw means
  // the node restarted and lost whatever config it had. Re-sync immediately in
  // that case rather than waiting out the timer.
  const prevBoot = await (async () => {
    const { data } = await supabase
      .from("sensor_logs").select("boot")
      .eq("device_id", device.id)
      .order("recorded_at", { ascending: false })
      .limit(1).maybeSingle();
    return typeof data?.boot === "number" ? data.boot : null;
  })();
  const thisBoot = typeof decoded.boot === "number" ? decoded.boot : null;
  const nodeRestarted = prevBoot !== null && thisBoot !== null && thisBoot <= prevBoot;

  const endDeviceIds = body?.end_device_ids as Record<string, unknown> | undefined;
  const ttnDeviceId = endDeviceIds?.device_id as string | undefined;
  // lora_slot_seconds resolves the shared value across a blower/filter pair
  // (see the migration): if the two halves are configured differently the
  // faster one wins for both, because slots that don't match don't share
  // boundaries and the pair would stop sampling together.
  const slot = device.lora_slot_seconds as number | null;
  const txOffset = (device.lora_tx_offset_seconds as number | null) ?? 0;

  if (ttnDeviceId && typeof slot === "number" && slot > 0) {
    const syncedAt = device.sample_slot_synced_at as string | null;
    const ageHours = syncedAt
      ? (Date.now() - new Date(syncedAt).getTime()) / 3_600_000
      : Infinity;

    if (ageHours >= RESYNC_HOURS || nodeRestarted) {
      const failure = await queueSlotDownlink(ttnDeviceId, slot, txOffset);
      // Stamp ONLY on success. Stamping regardless (what this did before)
      // made a failed POST indistinguishable from a delivered downlink:
      // devices showed "synced" while TTN had never been asked, and the
      // rationing then blocked retries for RESYNC_HOURS. Recording the
      // reason means the next failure is visible in one query rather than
      // inferred from nodes silently ignoring their slot.
      await supabase
        .from("devices")
        .update(failure
          ? { sample_slot_sync_error: failure }
          : { sample_slot_synced_at: new Date().toISOString(), sample_slot_sync_error: null })
        .eq("id", device.id);
    }
  }

  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
});