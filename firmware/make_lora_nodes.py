"""
firmware/make_lora_nodes.py
Generates one flashable sketch per LoRaWAN node from the single shared
source, differing ONLY in the TTN credentials block.

WHY. filtsure_lora_slot_sync.ino carries ONE node's DevEUI hardcoded, with a
comment telling you to edit it before flashing any other node. That edit is a
manual step performed immediately before a flash, and when it is missed the
result is the failure the file itself warns about: two physical nodes sharing
one DevEUI, TTN treating them as a single device, each join silently
invalidating the other's session, and MIC mismatches alternating between
units. No repo copy ever held the other nodes' values, so that edit was being
made and discarded every single time.

Generating the per-node files removes the edit. The shared body stays in one
place (the base sketch); only the credential block is substituted.

A node whose credentials are not known gets a literal #error in its sketch,
so it CANNOT be compiled until someone fills them in -- a placeholder that
silently compiled would just reproduce the original bug with extra steps. All
three nodes have real values as of 2026-10-13, so that path is currently
unused but kept for the next board.

ON THE P-NUMBER MAPPING, deliberately not encoded here. The base sketch
labels DevEUI ...772D as "P9 (blower)", while the bench spreadsheet lists that
DevEUI against PCB 4 / TTN test05. Those disagree, and nothing available
resolves it: devices.device_mac holds ESP32 MACs (E4B323FFFE8E, FC012CFFFEF9,
FC012CF905D4), which have no derivable relationship to a DevEUI. Since the
firmware does not need the P-number -- duct_role and the transmit offset are
assigned server-side from the MAC in the uplink payload -- these sketches are
keyed by the two identifiers the spreadsheet states unambiguously, and the
mapping is left to be confirmed rather than guessed.

APPKEYS LIVE IN firmware/lora_keys.json, NOT HERE. That file and the
generated lora_nodes/ sketches are gitignored: an AppKey is a node's root key,
and anyone holding it can join TTN as that node. This script, the base sketch
and lora_keys.example.json carry no key material, so the repo can be shared.
A node with no entry in lora_keys.json gets the #error block below, same as a
node with no DevEUI. On a new machine, copy lora_keys.json over (or fill in
the example from TTN Console -> End devices -> node -> Join settings).

Usage:
  python firmware/make_lora_nodes.py            # write the sketches
  python firmware/make_lora_nodes.py --check    # report, write nothing
"""
import argparse
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "filtsure_lora_slot_sync" / "filtsure_lora_slot_sync.ino"
OUT = HERE / "lora_nodes"
KEYS = HERE / "lora_keys.json"     # gitignored; see the docstring

# The block to replace: from the credentials banner up to (not including)
# the next section header. Matched rather than line-numbered so the
# generator keeps working if the body above it changes.
START = "/* ===================== TTN CREDENTIALS ====================="
END = "/* ===================== RADIO / SENSOR OBJECTS ====================="

# One entry per LoRa PCB, transcribed from the bench spreadsheet (2026-10-13).
# AppKeys are looked up by "sketch" in lora_keys.json.
# Keyed by TTN device name + PCB number, NOT by P-number: the screenshot gives
# DevEUI/AppKey against PCB and TTN name unambiguously, while the P-number
# mapping is contradictory (see the note in the module docstring). The firmware
# does not need the P-number anyway -- duct_role and the transmit offset are
# assigned server-side from the MAC in the uplink, not compiled in.
NODES = [
    {
        "sketch": "FILTSURE_LORA_P4",
        "label": "P4  (PCB 4, TTN device_id test05)",
        "mac": "DevEUI 0x70B3D57ED007772D",
        "role": None,
        "dev_eui": "0x70B3D57ED007772D",
    },
    {
        "sketch": "FILTSURE_LORA_P8",
        "label": "P8  (PCB 8, TTN device_id test04)",
        "mac": "DevEUI 0x70B3D57ED007771B",
        "role": None,
        "dev_eui": "0x70B3D57ED007771B",
    },
    {
        "sketch": "FILTSURE_LORA_P9",
        "label": "P9  (PCB 9, TTN device_id test03)",
        "mac": "DevEUI 0x70B3D57ED00766C3",
        "role": None,
        "dev_eui": "0x70B3D57ED00766C3",
    },
    {
        # The ex-WiFi spare (Supabase device_mac FC012CF906D4), re-fitted with a
        # LoRa radio and registered 2026-10-16. First created as "p5-lora", then
        # re-registered as "p13" to match the name the fleet uses -- safe only
        # because it had never joined, so no session was lost. DevEUI and AppKey
        # were carried across, so this sketch did not change.
        "sketch": "FILTSURE_LORA_P13",
        "label": "P13  (ex-WiFi spare, TTN device_id p13)",
        "mac": "DevEUI 0x70B3D57ED007985D",
        "role": None,
        "dev_eui": "0x70B3D57ED007985D",
    },
]

BANNER = '''/* ===================== TTN CREDENTIALS =====================
 * {label} -- {mac}{role_note}
 *
 * GENERATED FILE. Produced by firmware/make_lora_nodes.py from
 * firmware/filtsure_lora_slot_sync/filtsure_lora_slot_sync.ino. Everything
 * below this block is shared with every other node; edit the base sketch and
 * regenerate rather than editing here, or your change reaches one node only.
 *
 * This block is the ONLY thing that differs between nodes, and it is the
 * reason these files exist. devEUI identifies one physical node. Two nodes
 * sharing a DevEUI are one device to TTN: each join invalidates the other's
 * session, and whichever uplinks next fails its MIC check against keys the
 * network has already replaced -- intermittent, alternating, and very hard
 * to attribute, because each node looks fine on its own.
 */
'''

CREDS_OK = '''uint64_t joinEUI = 0x0000000000000000;
uint64_t devEUI  = {dev_eui};
uint8_t  appKey[] = {{ {app_key} }};
uint8_t  nwkKey[] = {{ {app_key} }};
'''

CREDS_MISSING = '''// ---------------------------------------------------------------------------
// FILL IN {label}'s OWN CREDENTIALS, THEN DELETE THE #error LINE BELOW.
//
// TTN Console -> Applications -> <your app> -> End devices -> this node:
//   DevEUI  is on the device overview page
//   AppKey  is under Join settings (reveal it there)
//
// Do NOT paste another node's values. That is the exact failure this file
// was created to prevent.
// ---------------------------------------------------------------------------
#error "{label}: DevEUI and AppKey not set. See the comment directly above this line."

uint64_t joinEUI = 0x0000000000000000;
uint64_t devEUI  = 0x0000000000000000;   // <-- {label}'s DevEUI
uint8_t  appKey[] = {{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }};
uint8_t  nwkKey[] = {{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }};
'''


def load_keys():
    """sketch name -> list of 16 "0x.." strings. Missing file is not an error:
    every node then generates with #error, which is the safe failure."""
    if not KEYS.exists():
        return {}
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    for name, key in keys.items():
        if len(key) != 16:
            raise SystemExit(f"{KEYS.name}: {name} has {len(key)} bytes, AppKey is 16")
    return keys


def build(base_text, node, app_key):
    i, j = base_text.index(START), base_text.index(END)
    role_note = f", duct_role {node['role']}" if node["role"] else ""
    head = BANNER.format(label=node["label"], mac=node["mac"], role_note=role_note)
    # An all-zero key is lora_keys.example.json copied without filling it in.
    if node["dev_eui"] and app_key and any(int(b, 16) for b in app_key):
        body = CREDS_OK.format(dev_eui=node["dev_eui"], app_key=", ".join(app_key))
    else:
        body = CREDS_MISSING.format(label=node["label"])
    return base_text[:i] + head + body + "\n" + base_text[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    base = io.open(BASE, encoding="utf-8").read()
    for marker in (START, END):
        if marker not in base:
            raise SystemExit(f"Base sketch no longer contains:\n  {marker}\n"
                             f"The generator matches on these headers; update it.")

    keys = load_keys()
    print(f"base: {BASE.relative_to(HERE.parent)}  ({len(base.splitlines())} lines)")
    print(f"keys: {KEYS.name} ({'%d node(s)' % len(keys) if keys else 'MISSING -- every sketch gets #error'})\n")
    for node in NODES:
        text = build(base, node, keys.get(node["sketch"]))
        folder = OUT / node["sketch"]
        dest = folder / f"{node['sketch']}.ino"
        ready = ("BLOCKED: #error until credentials filled in" if "#error" in text[text.index(START):text.index(END)]
                 else "ready to flash")
        print(f"  {node['sketch']:<30} {node['mac']}  {ready}")
        if args.check:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        # Path.write_text gained newline= only in 3.10; this runs on 3.9.
        with io.open(dest, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)

    if args.check:
        print("\n--check: nothing written.")
    else:
        print(f"\nwritten to {OUT}")
        print("Arduino requires folder name == sketch name; both are set.")


if __name__ == "__main__":
    main()
