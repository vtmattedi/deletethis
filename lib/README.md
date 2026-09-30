# Vendored libraries

Anything here is checked in rather than pulled from the PlatformIO registry,
because it is either patched, unpublished, or pinned to a version the project
depends on. Registry dependencies live in `platformio.ini` under `lib_deps`.

| Folder        | What it is                                                                                |
| ------------- | ----------------------------------------------------------------------------------------- |
| `boardstuff/` | Board support for the Sunton ESP32-3248S035C: ST7796 panel bring-up, GT911 touch, RGB LED, PWM backlight, and the LVGL display/input driver registration. Project-specific, not upstream. |
| `lvgl/`       | LVGL 8.3.6. Pinned: LVGL 9 changed the driver API that `boardstuff` registers against.     |
| `Time-master/`| The Arduino Time library (`TimeLib.h`), required by NightMareNetwork's `ServerVariable`.   |
| `lv_conf.h`   | LVGL's configuration. It sits beside `lvgl/` rather than inside it so the library folder stays a clean checkout; `platformio.ini` puts `lib/` on the include path and sets `LV_CONF_INCLUDE_SIMPLE` so LVGL finds it. |

## Notes

- **`Time-master` has no `Time.h`.** The upstream library ships one as a
  deprecated alias for `TimeLib.h`. It was deleted here on purpose: Windows
  filesystems are case-insensitive, so with `lib/Time-master` on the include
  path, a `#include <time.h>` anywhere in the build — the Arduino `HTTPClient`
  does exactly this — resolves to `Time.h` instead of the C standard header and
  fails to compile. Do not restore it.

- **`lv_conf.h` pins the colour format** to `LV_COLOR_DEPTH 16` with
  `LV_COLOR_16_SWAP 0`. The generated image assets in `src/UI/Assets/` and
  `include/Weather icons/` are emitted for exactly that format and `#error` if
  it changes.
