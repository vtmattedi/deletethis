#pragma once

#include <NightMare/HardwareProfile.h>

namespace NMHardware
{
inline Profile projectProfile()
{
    // Watson classic ESP32 wiring:
    //   GPIO26 -> INMP441 SCK/BCLK
    //   GPIO25 -> INMP441 WS/LRCLK
    //   GPIO33 <- INMP441 SD/DOUT
    //   3V3    -> INMP441 VDD
    //   GND    -> INMP441 GND
    //   GND    -> INMP441 L/R (LOW = left channel)

    static const Terminal esp32Terminals[] = {
        {"GPIO26"},
        {"GPIO25"},
        {"GPIO33"},
        {"3V3", CanonicalNet::V3v3},
        {"GND", CanonicalNet::Gnd},
    };

    static const Device esp32Devices[] = {
        {"mcu", esp32Terminals, 5,
         "ESP32", "mcu", "ESP32-WROOM-32", "Espressif"},
    };

    static const ConnectorContact esp32HeaderContacts[] = {
        {"GPIO26"},
        {"GPIO25"},
        {"GPIO33"},
        {"3V3", CanonicalNet::V3v3},
        {"GND", CanonicalNet::Gnd},
    };

    static const Connector esp32Connectors[] = {
        {"headers", esp32HeaderContacts, 5,
         "DevKit headers", ConnectorKind::Header},
    };

    static const Connection esp32InternalConnections[] = {
        {{"", EndpointKind::DeviceTerminal, "mcu", "GPIO26"},
         {"", EndpointKind::ConnectorContact, "headers", "GPIO26"}},
        {{"", EndpointKind::DeviceTerminal, "mcu", "GPIO25"},
         {"", EndpointKind::ConnectorContact, "headers", "GPIO25"}},
        {{"", EndpointKind::DeviceTerminal, "mcu", "GPIO33"},
         {"", EndpointKind::ConnectorContact, "headers", "GPIO33"}},
        {{"", EndpointKind::DeviceTerminal, "mcu", "3V3"},
         {"", EndpointKind::ConnectorContact, "headers", "3V3"}},
        {{"", EndpointKind::DeviceTerminal, "mcu", "GND"},
         {"", EndpointKind::ConnectorContact, "headers", "GND"}},
    };

    static const Terminal inmp441Terminals[] = {
        {"SCK"},
        {"WS"},
        {"SD"},
        {"VDD", CanonicalNet::V3v3},
        {"GND", CanonicalNet::Gnd},
        {"L/R"},
    };

    static const Device inmp441Devices[] = {
        {"mic", inmp441Terminals, 6,
         "Digital microphone", "microphone", "INMP441", "TDK InvenSense"},
    };

    static const ConnectorContact inmp441PinContacts[] = {
        {"SCK"},
        {"WS"},
        {"SD"},
        {"VDD", CanonicalNet::V3v3},
        {"GND", CanonicalNet::Gnd},
        {"L/R"},
    };

    static const Connector inmp441Connectors[] = {
        {"pins", inmp441PinContacts, 6,
         "INMP441 pins", ConnectorKind::Header},
    };

    static const Connection inmp441InternalConnections[] = {
        {{"", EndpointKind::DeviceTerminal, "mic", "SCK"},
         {"", EndpointKind::ConnectorContact, "pins", "SCK"}},
        {{"", EndpointKind::DeviceTerminal, "mic", "WS"},
         {"", EndpointKind::ConnectorContact, "pins", "WS"}},
        {{"", EndpointKind::DeviceTerminal, "mic", "SD"},
         {"", EndpointKind::ConnectorContact, "pins", "SD"}},
        {{"", EndpointKind::DeviceTerminal, "mic", "VDD"},
         {"", EndpointKind::ConnectorContact, "pins", "VDD"}},
        {{"", EndpointKind::DeviceTerminal, "mic", "GND"},
         {"", EndpointKind::ConnectorContact, "pins", "GND"}},
        {{"", EndpointKind::DeviceTerminal, "mic", "L/R"},
         {"", EndpointKind::ConnectorContact, "pins", "L/R"}},
    };

    static const HardwareDefinition definitions[] = {
        {"esp32-devkit-v1",
         AssemblyKind::MarketBoard,
         "ESP32 DevKit v1",
         "ESP32 DevKit V1",
         nullptr,
         {nullptr, 0,
          esp32Devices, 1,
          esp32Connectors, 1,
          esp32InternalConnections, 5}},

        {"inmp441-module",
         AssemblyKind::Module,
         "INMP441 microphone module",
         "INMP441",
         nullptr,
         {nullptr, 0,
          inmp441Devices, 1,
          inmp441Connectors, 1,
          inmp441InternalConnections, 6}},
    };

    static const Assembly watsonChildren[] = {
        {"controller", "esp32-devkit-v1", "ESP32 controller"},
        {"microphone", "inmp441-module", "INMP441 microphone"},
    };

    static const Connection watsonConnections[] = {
        {{"controller", EndpointKind::ConnectorContact, "headers", "GPIO26"},
         {"microphone", EndpointKind::ConnectorContact, "pins", "SCK"},
         {nullptr, nullptr, "I2S BCLK"}},

        {{"controller", EndpointKind::ConnectorContact, "headers", "GPIO25"},
         {"microphone", EndpointKind::ConnectorContact, "pins", "WS"},
         {nullptr, nullptr, "I2S WS/LRCLK"}},

        {{"controller", EndpointKind::ConnectorContact, "headers", "GPIO33"},
         {"microphone", EndpointKind::ConnectorContact, "pins", "SD"},
         {nullptr, nullptr, "I2S SD"}},

        {{"controller", EndpointKind::ConnectorContact, "headers", "3V3"},
         {"microphone", EndpointKind::ConnectorContact, "pins", "VDD"},
         {nullptr, nullptr, "3.3V"}},

        {{"controller", EndpointKind::ConnectorContact, "headers", "GND"},
         {"microphone", EndpointKind::ConnectorContact, "pins", "GND"},
         {nullptr, nullptr, "GND"}},

        {{"controller", EndpointKind::ConnectorContact, "headers", "GND"},
         {"microphone", EndpointKind::ConnectorContact, "pins", "L/R"},
         {nullptr, nullptr, "Left channel select"}},
    };

    static const Assembly roots[] = {
        {"watson",
         nullptr,
         "Watson acoustic device",
         AssemblyKind::Generic,
         "Watson",
         nullptr,
         nullptr,
         nullptr,
         {watsonChildren, 2,
          nullptr, 0,
          nullptr, 0,
          watsonConnections, 6}},
    };

    return {
        "watson/controller",
        definitions,
        2,
        roots,
        1,
        nullptr,
        0,
    };
}
} // namespace NMHardware
