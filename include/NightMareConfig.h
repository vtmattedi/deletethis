#pragma once

// NightMareNetwork feature selection for Watson.
//
// Watson is an acoustic observation device: it publishes two Sensors and one
// Event, and exposes its detector settings as Configs. It needs the Resource
// layer, a network connection, Wi-Fi (broker + OTA) and OTA. It does not need
// HTTP, WebSocket or LVGL.

#define NM_FIRMWARE_VERSION "watson-1.0.0"

#define NM_ENABLE_SETTINGS 1
#define NM_ENABLE_RESOURCES 1
#define NM_ENABLE_NETWORK 1
#define NM_ENABLE_CONSOLE 1
#define NM_ENABLE_WIFI 1
#define NM_ENABLE_MQTT 1
#define NM_ENABLE_TELEMETRY 1
#define NM_ENABLE_SCHEDULER 1
#define NM_ENABLE_JOBS 1
// The TLS broker connection validates certificates against the wall clock, so
// SNTP is required even though Watson itself only uses uptime.
#define NM_ENABLE_TIME_SYNC 1

// OTA is NightMareNetwork's, not a second project-specific ArduinoOTA.
#define NM_ENABLE_OTA 1
#define NM_ENABLE_HTTP 0
#define NM_ENABLE_WEBSOCKET 0
#define NM_ENABLE_LVGL 0

// Connection profiles. ESP-NOW (to the NightMare gateway) is preferred, with
// the remote (TLS) broker -- the one the PC backend and the controller already
// use -- as the failover when no gateway answers within
// nightmare:connection:failover_secs. There is no local broker.
//
// The Wi-Fi station only runs while an MQTT profile is selected, so while
// ESP-NOW is the active connection there is no IP link: SNTP, OTA and the raw
// PCM debug server wait for the failover to MQTT (or for an explicit
// `NETWORK SET MQTT`). Needs NM_ESPNOW_PSK in creds.h.
#define NM_NETWORK_MQTT 1
#define NM_NETWORK_LOCALMQTT 0
#define NM_NETWORK_ESPNOW 1

#define NM_CONSOLE_BUILTINS 1
// Serial carries the NightMare console (CONFIG ..., > resource ..., and the
// WATSON STATS command registered in main.cpp). It never carries audio.
#define NM_CONSOLE_SERIAL 1

// The Scheduler runs from tickNightMareESP() in loop() rather than in a task
// of its own: loop() spins every few ms, and a task stack is internal RAM that
// the TLS session needs.
#define NM_SCHEDULER_OWN_TASK 0

#define NM_LOG_LEVEL 2
