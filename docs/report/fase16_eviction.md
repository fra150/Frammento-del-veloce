# Fase 16 — Dimenticare apposta (eviction misurata)

n=40 N=16 T=0.05 seed=7

| politica | k | rimasti | copertura | Q rimasti | memoria | risparmio |
|---|---|---|---|---|---|---|
| eta | 0 | 40 | 1.000 | 0.9472 | 320 KB | 0 KB |
| eta | 10 | 30 | 0.750 | 0.9469 | 240 KB | 80 KB |
| eta | 20 | 20 | 0.500 | 0.9465 | 160 KB | 160 KB |
| eta | 30 | 10 | 0.250 | 0.9457 | 80 KB | 240 KB |
| q | 0 | 40 | 1.000 | 0.9472 | 320 KB | 0 KB |
| q | 10 | 30 | 0.750 | 0.9503 | 240 KB | 80 KB |
| q | 20 | 20 | 0.500 | 0.9531 | 160 KB | 160 KB |
| q | 30 | 10 | 0.250 | 0.9559 | 80 KB | 240 KB |
| uso | 0 | 40 | 1.000 | 0.9472 | 320 KB | 0 KB |
| uso | 10 | 30 | 0.750 | 0.9465 | 240 KB | 80 KB |
| uso | 20 | 20 | 0.500 | 0.9476 | 160 KB | 160 KB |
| uso | 30 | 10 | 0.250 | 0.9476 | 80 KB | 240 KB |

Eviction = cancellazione, mai sovrascrittura: i rimasti restano bit-identici (zero degrado sui non evictati).
