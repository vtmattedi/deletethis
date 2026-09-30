# NightMare Dashboard

A wall panel for the NightMare network. It owns no hardware of its own: it joins
the network as an ordinary NightMare device and drives an air conditioner, a
light and an RGB light that live on *other* devices, over the NightMare resource
protocol.

Hardware: **Sunton ESP32-3248S035C** — ESP32-WROOM-32, 4MB flash, no PSRAM,
ST7796 320×480 SPI panel, GT911 capacitive touch, RGB status LED, PWM backlight.
Mounted **landscape**: the UI is 480×320.

## Screens

Status bar across the top, nav bar across the bottom, pages in between — 464×246
of content. That makes height the scarce dimension and width the plentiful one,
so most pages are two columns.

**The AC and light cards are switches.** Tapping the card toggles the thing it
represents and its colour is the state; anything finer sits on a nested button,
which LVGL does not bubble, so pressing one does not also toggle the card.

**Unbound or offline controls stay on screen, greyed and inert**, rather than
being replaced by a notice — the page keeps its shape, and what the panel would
let you do stays visible. A line on the card says which it is ("nothing bound"
or "<device> is not answering"). Offline means the panel has heard no retained
`online` status from that device. Tapping an unbound AC card jumps to Devices;
tapping an unbound light card jumps to Resources, which is where a light is
chosen.

Two implementation notes for anyone extending this. The greying is done with
explicit colours, not `LV_STYLE_OPA`: any opacity below 255 makes LVGL render
that card into an intermediate layer, asking for up to `LV_LAYER_SIMPLE_BUF_SIZE`
(24KB) on *every redraw*, which this board cannot spare. And LVGL still delivers
clicks to a button in `LV_STATE_DISABLED`, so every handler re-checks liveness
itself rather than trusting the greyed-out look.

| Screen       | What it does                                                                 |
| ------------ | ---------------------------------------------------------------------------- |
| **Home**     | Left column, the AC: tap to toggle power, with state, setpoint, door, and buttons for Auto, sleep (`🔔`) and adjust (`⚙`). Adjust expands over the page for setpoint −/+ and **Sync with unit**. Right column (scrolls): weather (tap to refresh), the light (tap to toggle) and the RGB light. |
| **Devices**  | Everything heard announcing itself on the broker, with online state and age. Tap a row for that device's detail page. |
| **Device**   | Header spans the page; below it, left: bind as the air conditioner, reboot or delete. Right: Boot / Hardware / System across the top, with the device's answer under them. |
| **Resources**| Every resource value seen on the network, newest per address, with a badge on anything already in use. Tap a row for that resource's page. |
| **Resource** | One resource: how long ago it arrived, its device, its full topic and its value — and the buttons that give it a job. Light, RGB colour, door sensor (1 = open / 1 = closed), resting screen, or the whole device as the air conditioner. The button for whatever it already does is lit. |
| **Setup**    | Left: brightness, resting brightness, idle timeout — applied and saved as they move — and **Rest now**. Right: name, firmware, IP, WiFi, broker, uptime, heap; switch brokers, request MessagePack manifests from every device, or reboot the panel. |
| **Resting**  | After the idle timeout or Rest now: a dimmed logo and a 120px clock with a blinking colon, with the corners carrying what matters — top-left anything wrong (door open, no WiFi, no broker), top-right temperatures, bottom-left your chosen resource, bottom-right what is switched on. Any touch wakes it. |

## Build and flash

```sh
pio run                 # build
pio run -t upload       # flash over USB
pio device monitor      # serial console, 115200
```

First build only:

```sh
cp include/creds.example.h include/creds.h   # then fill it in
```

`include/creds.h` holds the WiFi and MQTT passwords and is gitignored.
`include/NightMareConfig.h` selects the library's features and is not secret —
it is where the timezone, the log level and the threading choices live.

## What the panel says about itself

The panel is an ordinary NightMare device, so it publishes the same retained
documents it reads from everything else: `<panel>/status`, `<panel>/info`, and
`<panel>/telemetry/{system,network}`. Its hardware topology is retained as
`<panel>/hardware` and `<panel>/hardware/msgpack`; `HW` returns the JSON form
over the console and can republish either encoding.

[include/NightMareHardware.h](include/NightMareHardware.h) is what makes
`hardware.board` read `sunton-esp32-3248s035c:v1` and the topology lists all
eighteen pins instead of `unspecified` / `none`. Connections reference compact
device and bus indexes and carry signal type, direction, pull, and active-low
state. Pin-specific operating details remain in the source comments rather
than being sent as rendering metadata.

Those numbers are **copied** from `lib/boardstuff/boardstuff.h` rather than
included from it, and the two have to be changed together. The header is read
by the *library's* build, and `boardstuff.h` pulls in LVGL and Wire — including
it would drag the whole display stack into NightMareNetwork's translation units
to fetch a dozen integers. The board is fixed hardware and these numbers change
about never; that is the trade, and it is written at the top of the file.

This list used to be expensive: `INFO` sized its JSON pool as
`2048 + connections * 256`, so eighteen pins meant a single contiguous ~6.4KB
allocation on every MQTT connect. Under ArduinoJson 7 the document grows
through a chain of small pools instead, so the length of this list no longer
decides the size of any one allocation.

## How it talks to the network

Every capability on a NightMare network has an address of the form

```
<device>/resource/<resource>
```

with retained manifests at `<device>/manifest` and
`<device>/manifest/msgpack`, plus resource topics under
`<device>/resource/<resource>`:
retained value at `…/state`, and the transient `…/set` and `…/invoke`. The
governing rule is **a manifest describes, `/state` tells the truth** — a missing
or disagreeing manifest never invalidates a value.

The panel uses that in two different ways, and the split is the shape of the
whole application.

### What it drives

Everything the panel controls is a **bound Remote resource**. The library owns
the subscription, the decoding, the freshness and the optimistic write; the
panel only says what each one is pointed at.

- [src/App/AcClient.cpp](src/App/AcClient.cpp) — the air conditioner's nine
  resources.
- [src/App/LightControl.cpp](src/App/LightControl.cpp) — the light and the
  colour.

Each is declared with no source and pointed at whatever is bound by
`…_applyTargets()`, called from `Net_loop()` when a binding changes. Retargeting
is a `setSource()`, not a rebuild: the library unsubscribes the old address,
drops everything it learned from it, subscribes the new one, and the broker
replays that device's retained state on its own.

Writes are **optimistic**. Tapping power flips the card immediately; the owner's
retained state wins again when it arrives, or when the ~5s window closes. So a
control that appears to work and then reverts means the command did not take,
not that the panel dropped it.

### What it watches

The library no longer subscribes to `#`, so the Devices and Resources screens
are fed by three explicit subscriptions installed in
[src/App/Net.cpp](src/App/Net.cpp). They are remembered in RAM and reinstalled
on every reconnect.

| Subscription            | Goes to                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `+/status`              | Device registry. Retained JSON: `{"name":…,"hardware":…,"online":true}`. An empty payload is a tombstone — it takes a device offline but never creates one. |
| `+/resource/+/state`    | Resource registry, one entry per `(device, resource)`, holding the codec representation exactly as published. |
| `Control/forecast`      | The weather model.                                             |

Anything the resource manager recognises as belonging to a *bound* resource is
consumed before it reaches the panel's own callback — which is why
`LightControl_mirrorToRegistry()` copies the bound light and colour back into
the registry once a second. Without it a light would vanish from the Resources
screen the moment it was given a job, taking the only way to unbind it with it.
The AC's resources are deliberately *not* mirrored: they are all on the Home
card already, and eighteen of them would crowd every other device out of a
40-entry table.

`Control/time` never reaches the panel's code at all — the library consumes it
inside the MQTT client and sets the clock — and time sync needs no request job
of its own, because the library runs SNTP and asks on `Control/request` whenever
its clock is invalid.

## Binding

The panel starts bound to nothing and says so on each card. The two registers
are bound in different places, because they are different shapes of thing.

**The AC is bound to a device**, on the **Devices → Device** page. Its nine
resources are a contract the controller publishes under names it chose, so
naming the device names all of them:

| Resource            | Type          | What the panel does with it |
| ------------------- | ------------- | --------------------------- |
| `ac_state`          | int8 sensor   | the state chip, and the card's freshness |
| `ac_known`          | bool sensor   | shows "unknown" when the controller does not believe itself |
| `ac_sleep_deadline` | uint32 sensor | the sleep countdown, as an epoch second |
| `temperature`       | float sensor  | the room reading it controls against |
| `ac_power`          | bool state    | the card, and tapping it |
| `ac_temperature`    | uint8 state   | the unit's own setpoint, 18–30 |
| `ac_target`         | float state   | the room target; negative disables the thermostat |
| `ac_manual_sync`    | action        | `{"power":bool,"temperature":int}` |
| `ac_sleep`          | action        | `{"minutes":int}`, 0 cancels |

**The light and the colour are bound to one resource each**, from a resource's
own page — **Resources → tap a row**. Nothing in the protocol says a light must
be called `light`, so rather than assuming, the panel shows what is actually
being published and asks what it is for. The door sensor and the resting
screen's free slot are chosen the same way. That page also offers the AC, as a
shortcut for someone standing on `ac_power`; it binds the device, and the button
says so.

**A value that is a JSON document reads as `struct` in the list**, and only
there. A status document clipped to a 90px column shows something like
`{"state":1,"tar`, which is worse than useless — the type is the honest summary,
and the resource page flattens the whole thing to `key  value` lines one tap
away. That test is the *shape* of the payload, not the protocol's `struct` value
type: the panel never reads manifests, so it has no declared type to go on. A
device publishing JSON under a `string` type — which is what an AC controller's
`ac_status` is — reads as a struct here, and for the question being asked ("will
this fit in a row") that is the right answer.

Roles are **not exclusive by device**. One board can be the air conditioner and
own the bound colour as well — that is a real device, not a mistake. The only
thing that cannot be shared is a single `(device, resource)` address, which the
resource manager refuses on its own.

Bindings persist in the NightMare settings store (`ac_host`, `light_host` /
`light_res`, `rgb_host` / `rgb_res`), so they survive a reboot but not a
filesystem wipe. They can also be set from the console — over serial, or over
MQTT on `<panel>/console/in`:

```
TARGET                          # show current bindings as JSON
TARGET AC Adler                 # bind the AC role to the device "Adler"
TARGET LIGHT Mycroft light      # bind the light role to Mycroft's "light"
TARGET RGB Sherlock             # bind the colour role, default resource name
TARGET LIGHT ""                 # unbind
FORECAST                        # request a forecast now
HEAP                            # free / min-ever-free / largest free block
```

Names are **case-sensitive** — `Micro` and `micro` are different devices.

A `TARGET` that changes something is *staged*, not applied: a command arriving
over MQTT runs on the MQTT client's task, and the bindings belong to `loop()`.
`Net_loop()` picks it up within a few milliseconds, so the effect is immediate,
but the reply reports what was asked for rather than what is bound.

The framework's own built-ins (`PING`, `INFO`, `TIME`, `JOB`, `MQTT`, `WIFI`,
`CONFIG`, `REBOOT`) and the `>` resource-command grammar work on this panel too:
`>list` names every resource it has bound, and `> ac_power set true` drives one
directly.

### Brightness, out of one number

A colour resource is a single 24-bit value, so "off", "which colour" and "how
bright" all have to live in it. The panel reads them back out as: **off** is a
published 0, **brightness** is the largest of the three channels, and **colour**
is the value scaled back up so that channel reads 255. The hue and level last
chosen are kept on the panel as well, so switching off and on again restores
what it was, and the colour wheel opens on the colour that was picked rather
than on a dimmed version of it.

### Exploring a device

The detail page can ask a device about itself over MQTTP — the same
request/response-over-MQTT protocol the backend uses:

```
request  ->  <Device>/console/controlled/<id>/in     the command text
reply    <-  <Device>/console/controlled/<id>/out
```

Three commands are offered — `INFO BOOT`, `INFO HARDWARE`, `INFO SYSTEM` —
because those are sections of the framework's own telemetry document and every
NightMare device answers them without a project resolver. Nothing
device-specific is offered: a command implemented only in one firmware's local
resolver would simply time out everywhere else.

**The panel is a viewer, not a store.** That is the whole design. The backend
persists every answer from every device forever in Postgres; the panel holds one
answer, for one device, for as long as the page is open. Leaving frees it and
reopening asks again, which costs nothing because the data is live anyway.
Concretely, in [src/App/Mqttp.cpp](src/App/Mqttp.cpp):

- **One request in flight** — a single slot, not a map. One screen, one user,
  one question at a time, so memory is bounded by construction.
- **A fixed `.bss` buffer**, never grown, never freed. Reassembling chunks into
  a growing `String` is exactly the heap fragmentation that breaks the next
  mbedTLS handshake on this board — see the heap note below.
- **Truncate, don't fail** at `MQTTP_RESPONSE_MAX`. A partial answer with a
  marker beats an error.
- **8s timeout**, not the backend's 20s: someone is standing in front of it.

**Listening is per device and per page, not per request.** The library no longer
takes `#`, so replies have to be subscribed for — but subscribing *per request*
does not work, and this is worth knowing before anyone tries it again:
`esp_mqtt_client_subscribe()` only queues a SUBSCRIBE and returns, and at QoS 0
a device answering `INFO BOOT` in about a millisecond beats the broker's SUBACK
over a TLS link every time. The reply is dropped before anyone is listening for
it, and every query times out.

So `Mqttp_listenTo()` takes `<device>/console/controlled/+/out` when the device
page opens and gives it back when the page is left. The wildcard is one device
wide, the SUBACK has the time between opening a page and tapping a button, and
the request id still does the correlating. Only the synchronous variant is
implemented, not `asynccontrolled`.

Replies of 512 bytes or less arrive whole; larger ones are chunked by the sender
as `;;<n>/<total>;;<data>` and reassembled. Note that `MQTT_Send` drops empty
payloads, so a command that returns nothing is indistinguishable from a timeout.

### Rebooting and deleting a device

The same dialog offers one action on the device itself, chosen by whether it is
answering — the two are never both useful, so only one is shown:

- **Online → Reboot.** Publishes `REBOOT` to `<device>/console/in`.
- **Offline → Delete.** Forgets it locally, unbinds every role it held, and
  clears the retained announcements with zero-length retained publishes: its
  `status`, its `resources` manifest, and the `state` of every resource the
  panel has seen it publish. Any one of those left retained re-announces the
  device on the next resubscribe.

Both need a second tap to confirm. Delete does not decommission anything: a
device that is still running reappears on its next message. It clears the
leftovers of something already gone. A resource the panel never heard cannot be
tombstoned — the same limitation the library's own identity cleanup has, for the
same reason.

### The AC's door and unit sync

**Door.** Any boolean resource can be the door sensor: tap its row on Resources
and choose whether a reading of 1/true means *open* or *closed* — reed switches
report either, depending only on wiring. Only `true`/`false`/`1`/`0` are
understood; anything else shows as "no reading" rather than guessing. This is
independent of the AC device's own door logic, which still drives the
"paused - door" state. The binding is stored under `door_device` / `door_key`,
names kept from before the resource protocol so an already-deployed panel does
not lose them on the upgrade. The AC controller uses the same two names for its
own door reference, but in its own settings store on its own device — setting
one here does not set the other.

**Sync with unit.** Not a command to the AC. It tells the controller what the
physical unit is *already* doing, after it was changed behind the controller's
back — usually with the unit's own IR remote. Set the temperature and on/off the
unit's display shows, then Sync; the panel invokes `ac_manual_sync`. No
confirmation, because it corrects a belief rather than changing the room.

**Sleep is real now.** The `🔔` button offers Off / 30 min / 1 / 2 / 4 hours and
invokes `ac_sleep`; the controller owns the timer, counts it down itself and
publishes the deadline on `ac_sleep_deadline`, which is what the card displays.
The panel need not be awake for the unit to switch off. (This replaces a mock
that was kept and counted down on the panel only.)

## Other config keys

| Key          | Default | Meaning                                  |
| ------------ | ------- | ---------------------------------------- |
| `rest_after` | `60`    | Idle seconds before the resting screen. `0` disables it. |
| `bl_active`  | `200`   | Backlight level while in use, 8–255.     |
| `bl_rest`    | `24`    | Backlight level while resting, 0–255.    |
| `door_device` / `door_key` | — | Device and resource publishing the door state. Set from the Resources page. |
| `door_invert`| `0`     | `1` when a reading of 1/true means *closed* rather than open. |
| `rest_device` / `rest_key` | — | The resource shown in the resting screen's free corner. |
| `ac_timezone`| —       | POSIX TZ string. Overrides `NM_TIMEZONE` at runtime. |

The display keys are editable on the **Setup** page, which applies them
immediately. They can also be set over the console — `CONFIG SET rest_after 120`
— but the screen manager caches them, so a console change takes effect on the
next reboot unless something calls `Ui_reloadSettings()`.

The broker switch on Setup is deliberately *not* persisted: which broker is
reachable depends on where the panel is, not on the panel. Nothing switches it
back on its own — the library has no automatic broker failover.

## Backend: the forecast reply is not implemented yet

Time sync works today, and no longer needs the backend at all: the library runs
SNTP, and `Control/request` = `time` is only a fallback.

**Weather does not**: the panel publishes `Control/request` = `forecast` and
nothing replies, so the weather card stays on "waiting for forecast".

To finish it, add a `forecast` branch alongside the existing `time` one in
`backend/src/app/Services/ControlService.ts`, publishing to `Control/forecast`:

```json
{ "icon": "partly-cloudy-day", "temp": 24.1, "feels": 25.0,
  "min": 19.2, "max": 29.8, "rain": 20, "condition": "Partly cloudy" }
```

`icon` is a [Visual Crossing](https://www.visualcrossing.com/resources/documentation/weather-api/defining-icon-set-in-the-weather-api/)
icon name, mapped to the bundled amCharts art by `visualCrossingToId()` in
`include/Weather icons/icons/convert.h`. Every field is optional, and the longer
spellings `feelslike`, `tempmin`, `tempmax`, `precipprob`, `conditions` are
accepted too — so a Visual Crossing `currentConditions` object can be forwarded
more or less unchanged.

## Layout

```
src/
  main.cpp              Startup, the loop, and the TARGET/FORECAST/HEAP commands
  App/
    Net.cpp             The wildcard subscriptions, the inbox, message routing
    AcClient.cpp        The air conditioner's resource contract
    LightControl.cpp    The light and the colour, and the one-number colour model
    Targets.cpp         Which (device, resource) fills which role, persisted
    Registry.cpp        Bounded snapshot of devices and resource values seen
    Mqttp.cpp           Request/response over MQTT, one at a time, fixed buffer
    SensorBindings.cpp  The door and resting-screen resources
    Forecast.cpp        Weather model; the only place the icon set is included
  UI/
    Ui.cpp              Screen manager, nav bar, idle/rest transitions
    Theme.cpp           Colours, shared styles, widget builders
    Screen*.cpp         One file per screen; ScreenResource.cpp is the binder
    Assets/logo_mw.c    Generated by tools/png2lvgl.py
    Assets/font_clock_120.c  Montserrat 120px digits (SIL OFL), lv_font_conv; command in its header
include/
  NightMareConfig.h     Library feature selection; read this before changing behaviour
  NightMareHardware.h   The board and its pins, as published in <device>/info
  creds.h               WiFi and MQTT secrets; gitignored
lib/                    Vendored libraries -- see lib/README.md
tools/png2lvgl.py       PNG to LVGL C array converter
```

Screens build their widgets once and update them in place; nothing is created or
destroyed while the panel runs, so the heap does not fragment over long uptimes.

## Things to know

- **The panel is single-tasked, and two settings keep it that way.**
  NightMareNetwork calls the message callback from inside the esp-mqtt event
  handler, on a task that runs in parallel with `loop()` on this dual-core chip.
  The callback in [src/App/Net.cpp](src/App/Net.cpp) therefore only filters by
  topic, copies the message into a fixed-size ring buffer and returns;
  `Net_loop()` does all the routing. The queue drops rather than blocks when
  full, since a blocked MQTT task misses keep-alives; `Net_droppedMessages()`
  (and the `HEAP` command) should stay at 0.

  `NM_SCHEDULER_OWN_TASK 0` in `include/NightMareConfig.h` is the other half:
  with the default of 1 the Scheduler gets its own FreeRTOS task and every job
  callback — the forecast refresh, the heap log — would run in parallel with the
  screen it is updating. At 0, `tickNightMareESP()` runs them from `loop()`.

  Two things deliberately escape the rule, and both are safe for a stated
  reason. The library decodes bound-resource state on the MQTT task and stores
  it in the resource object — which is why every value the panel binds is a
  **scalar**: a bool, an int, a float, a 32-bit colour, so a read from `loop()`
  is a single aligned load and cannot tear. (`AcClient` deliberately does not
  consume the controller's `ac_status` String document for exactly this reason.)
  And a console command arriving over MQTT runs on the MQTT task, so a `TARGET`
  that changes a binding is staged through `Targets_request()` and applied by
  `Net_loop()` instead of being written there.

- **An LVGL out-of-memory restarts the panel; it does not freeze it.** LVGL's
  default assert handler is `while(1);`, which hung the `loop()` task — touch
  and Serial dead while the MQTT tasks kept logging. `LV_ASSERT_HANDLER` in
  `lib/lv_conf.h` now calls `lvgl_assert_failed()` (in `lib/boardstuff`), which
  prints `[lvgl] assert failed: free=… largest=… min=…` and restarts. The malloc
  assert stays enabled on purpose: about a quarter of LVGL's call sites do not
  check for NULL afterwards, so without it a failed allocation crashes somewhere
  arbitrary instead.

- **Heap is the tight resource, not flash.** No PSRAM, so LVGL widgets, the
  render buffer, the WiFi driver and mbedTLS all share ~290KB. mbedTLS needs
  tens of KB *contiguous* for each MQTT handshake, and when it cannot get it the
  IDF client does not fail cleanly — `esp_mqtt_client_init()` returns a
  half-built handle and the first publish panics the core into a boot loop.

  Three things keep that from happening, and all three are load-bearing:
  device rows are built **lazily**, while Resources uses four reusable rows and
  pages through the registry (neither allocates below the `Ui_canAllocateRow()`
  floor); the LVGL render buffer is small and deliberately **not**
  DMA-capable, since the DMA pool is what the WiFi driver draws from; and
  `startNightMareESP()` runs from `loop()` behind a heap check rather than from
  `setup()`.

  That last one moved. The library now brings MQTT up from inside the WiFi task
  on the first connection, so the panel can no longer guard the handshake
  directly — the last point it still controls is whether WiFi starts at all, and
  that is what `tryStartNetwork()` in [src/main.cpp](src/main.cpp) gates. It is
  a weaker guarantee, because association itself costs heap in between, but it
  is still the difference between a red MQTT icon and a reboot loop.

  Dialogs are built at boot, never on first tap: built on demand they ask for
  memory at whatever moment the tap lands, including mid-TLS-reconnect, and a
  failed allocation inside LVGL 8 crashes instead of failing. List rows are the
  opposite — built on demand and freed when their page is left or the panel
  rests. The Resources list never holds more than one four-row page, regardless
  of how many entries the registry contains.

  `HEAP` on the console reports free, minimum-ever-free, largest free block and
  dropped messages. Largest block matters as much as the total: a fragmented
  heap fails the handshake while looking comfortable — 91KB free with a 37KB
  largest block is a heap that cannot reconnect.

  The serial log samples it at `boot`, `after lvgl` and `after ui`, then at
  `mqtt up` and `after burst`, then `periodic` every 30 s. The middle pair
  brackets the worst moment on purpose, because the 30 s sample is far too
  coarse to catch it: between those two readings the broker replays every
  retained status and resource state on the network at once, and the library
  publishes this device's status, manifest, `/info` and both telemetry
  documents back to back, all while mbedTLS still holds the two 16KB record
  buffers from the handshake. An `esp_tls_conn_write error, errno=11` (EAGAIN,
  which the ESP-IDF strerror renders as the misleading "No more processes")
  followed by `mqtt_client: Error to resend data` is what running out of room
  there looks like from the outside: lwIP could not get a send buffer.

  **The numbers here are `MALLOC_CAP_8BIT`, not `ESP.getFreeHeap()`** — see
  [src/App/Heap.h](src/App/Heap.h). The Arduino accessors report
  `MALLOC_CAP_INTERNAL`, which counts IRAM that no byte-buffer `malloc` can
  ever hand out; on this board that reads about 37KB high and its
  largest-free-block never moves off 36852. Every gate that used to read them
  was measuring the wrong pool, and `Ui_canAllocateRow()` was returning true
  unconditionally as a result. The `(internal=…)` figure is still printed for
  comparison, because it is what every other ESP32 tool shows and the gap
  between the two is the point.

- **No OTA.** The firmware is ~2.1MB and a 4MB flash cannot hold two OTA slots
  that size — the partition table ([src/OneAppNoOta.csv](src/OneAppNoOta.csv))
  is a single ~3.75MB app. Flashing is over USB. Fitting OTA would mean cutting
  roughly 250KB, most easily from the 21 bundled weather icons (~12KB each) or
  the 180px logo (~97KB).

  `NM_ENABLE_OTA` is therefore 0 in `include/NightMareConfig.h`. Left on, it
  still starts ArduinoOTA on first connect — a task, an mDNS responder and a UDP
  listener, held forever — and that heap is what the HiveMQ TLS handshake runs
  short of. Turn it back on only together with an OTA-capable partition table.

- **The HiveMQ TLS connection is the single largest memory cost.** The prebuilt
  Arduino libraries fix mbedTLS at `SSL_MAX_CONTENT_LEN 16384` with dynamic
  buffers off and `KEEP_PEER_CERTIFICATE` on: every connection holds a 16KB input
  and a 16KB output record buffer (each one contiguous block) plus the broker's
  certificate chain, and a handshake needs roughly 50–60KB. A handshake that
  cannot get it fails with `esp-tls 0x801A` / `tls stack 0x2880`
  (`MBEDTLS_ERR_X509_ALLOC_FAILED`). None of it can be changed from
  `platformio.ini`. The two real fixes are the planned local broker (plain MQTT,
  no TLS) or rebuilding the framework libraries with
  `CONFIG_MBEDTLS_DYNAMIC_BUFFER` via pioarduino's `custom_sdkconfig`.

- **The platform pin matters.** `platform = espressif32` resolves to the
  pioarduino fork (55.x, Arduino core 3.x). Core 3.x is *required*:
  NightMareNetwork's MQTT module uses the nested `esp_mqtt_client_config_t` that
  only exists from ESP-IDF 5.0. Core 2.x will not compile.

- **If the display or touch is wrong on real hardware**, these are the knobs —
  none of it could be verified without the board in hand:
  - Colours inverted → add `-D TFT_INVERSION_ON` to `build_flags`.
  - Red and blue swapped → add `-D TFT_RGB_ORDER=1`.
  - Image upside down → in `platformio.ini`, swap `-D TFT_ORIENTATION_LANDSCAPE`
    for `-D TFT_ORIENTATION_LANDSCAPE_INV` (rotation 1 ↔ 3, same axis, 180°
    apart). Touch is derived from the same rotation, so it follows
    automatically — there is no second place to edit.
  - Image right but touch wrong → the two have genuinely drifted, and the
    rotation-1/3 branches of `gt911_read_touches()` in
    [lib/boardstuff/boardstuff.cpp](lib/boardstuff/boardstuff.cpp) are where to
    look. Portrait rotation 2 was verified on hardware; the landscape branches
    follow the same derivation but have not been.
  - Anything laying out pixels must use `SCREEN_WIDTH`/`SCREEN_HEIGHT`, never
    `TFT_WIDTH`/`TFT_HEIGHT`: those stay the panel's native portrait size and
    are what the touch transform is expressed in.

## License

This project is MIT licensed — see [LICENSE](LICENSE).

It also carries vendored third-party work, each under its own terms. Both are
redistributed here with their licence files intact, and one of them has a
condition you must keep:

| What | Where | Licence |
| --- | --- | --- |
| LVGL 8.3.6 | [lib/lvgl](lib/lvgl) | MIT, © LVGL Kft — [LICENCE.txt](lib/lvgl/LICENCE.txt) |
| Weather icons | [include/Weather icons](include/Weather%20icons) | **CC BY 4.0**, © [amCharts](https://www.amcharts.com/) — [LICENSE](include/Weather%20icons/LICENSE) |
| Font Awesome 5 webfont | `include/Weather icons/animated/c/` | Font: SIL OFL 1.1; icons: CC BY 4.0, © Fonticons, Inc. |

**The icons require attribution.** CC BY 4.0 is not public domain: anything that
ships these icons — this firmware, a fork, a photo of the panel used
commercially — has to credit amCharts. That obligation travels with the
artwork and is not waived by this project's MIT licence, which covers the code
only.

LVGL is vendored rather than pulled as a dependency because
[lib/lv_conf.h](lib/lv_conf.h) is written against that exact version; see
[lib/README.md](lib/README.md).
