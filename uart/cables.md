🔧 How to Use for UART Wiring
| Cable                                   | Conductor Size | Temperature Range     | Notes                                                                            |
| --------------------------------------- | -------------- | --------------------- | -------------------------------------------------------------------------------- |
| Norden 1173C (22 AWG)               | 22 AWG         | –20 °C to +80 °C  | PVC insulation, flexible, suitable for general indoor environments               |
| Maker Store Shielded 3 Core (7/0.5) | \~20–22 AWG    | –30 °C to +105 °C | Higher-temp rating, more rugged jacket, suitable for slightly harsher conditions |
| RS PRO 3-Core 24 AWG Screened       | 24 AWG         | –15 °C to +80 °C  | Standard for control/signal applications, PVC jacket                             |
 
| Cable Type       | Jacket    | Temp Range       | Notes                              |
| ---------------- | --------- | ---------------- | ---------------------------------- |
| PTFE/Teflon      | PTFE      | –200°C to +200°C | Flexible, stable, common in labs   |
| FEP or PFA       | FEP/PFA   | –90°C to +200°C  | Slightly stiffer, very durable     |
| Kapton-insulated | Polyimide | –269°C to +250°C | Expensive, used in vacuum or space |
| Axon Cryoflex®   | Hybrid    | Down to –269°C   | Engineered for ultra-low temps     |

Wiring advice:
Wire 1 → TX signal
Wire 2 → RX signal
Wire 3 → Ground (connect directly between devices)
Shield → Grounded at one end only (typically the Raspberry Pi side)