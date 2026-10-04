#pragma once

// ============================================================
// Copy this file to include/creds.h and fill it in:
//
//     cp include/creds.example.h include/creds.h
//
// creds.h is gitignored. This template is not, so keep the two
// in sync when adding a define.
// ============================================================

// --- NightMareNetwork (see its docs/getting-started.md) ---
#define MQTT_CREDS_H

#define DEFAULT_SSID     "your-network"
#define DEFAULT_PASSWORD "your-password"

// Remote (TLS) broker. No "mqtts://": NightMareNetwork builds the scheme.
#define REMOTE_MQTT_URL  "your-broker.example.com"
#define REMOTE_MQTT_PORT 8883

// Not used while NM_NETWORK_LOCALMQTT is 0 (NightMareConfig.h), but the
// MQTT driver still names them.
#define LOCAL_MQTT_HOST  "127.0.0.1"
#define LOCAL_MQTT_PORT  1883

#define MQTT_USER   "username"
#define MQTT_PASSWD "password"

// Root certificate of the broker's CA, PEM.
// A macro, not a variable: NightMareNetwork tests it with #ifndef.
#define ROOT_CA R"EOF(
-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----
)EOF"

// --- Watson ---
#define WIFI_SSID     DEFAULT_SSID
#define WIFI_PASSWORD DEFAULT_PASSWORD

// TCP port the raw PCM debug server listens on (ENABLE_TCP). The PC tools
// default to this, so there is usually no reason to change it.
#define AUDIO_TCP_PORT 3333
