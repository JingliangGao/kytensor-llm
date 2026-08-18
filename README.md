# llama.cpp 

branch : dev-xdl

# Hardware
| Category        | Item                    | Value                              |
| --------------- | ----------------------- | ---------------------------------- |
| Identity        | Device Name             | RPP_R9                             |
|                 | MFG_ID                  | XA092Y1610164                      |
|                 | Serial Number (SN)      | PK150000QTQXDL1A067U000000100000   |
|                 | Board Version           | A9_V2                              |
| Boot / Firmware | Boot Mode               | EEPROM / WORKING                   |
|                 | Firmware                | 20260709                           |
|                 | MCU                     | v0.3.2.1                           |
| PCIe            | PCI id                  | 0000:03:00.0                       |
|                 | PCIe Link Cap           | Gen3x4                             |
|                 | PCIe Link Status        | Gen3x4                             |
| Clock / Memory  | RPP Core                | 920MHz                             |
|                 | VE                      | OFF                                |
|                 | DDR Speed               | 4000 Mbps                          |
|                 | DDR Number              | 4                                  |
|                 | DDR Size                | 16GB                               |
| Thermal         | Fan Working Temp        | 80 C                               |
|                 | Clock Throttling Temp   | 85 C                               |
|                 | Trip Temp               | 104 C                              |
| Runtime Status  | FAN                     | 0%                                 |
|                 | TEMP                    | 62.08 C                            |
|                 | VOL                     | N/A                                |
|                 | RPP                     | 0.00%                              |

# 一键式配置环境、编译、测试脚本
```bash
chmod +x build-for-debug.sh
./build-for-debug.sh
```

