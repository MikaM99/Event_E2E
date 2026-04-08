Dokumentation...
Beginn der MA anhand der Auswahl eines passenden E2E-Modells - Wahl für CIL (conditional imitation learning)
- Modell lernt Fahrverhalten durch Nachahmung eines menschlichen fahrers
- Eingaben: Sensordaten, Fahrzusatnd und Navigationsbefehl
- Ausgaben: Steering, Gas, Brake
- Conditional = Navigationswerte als Bedingung: links abbiegen, rechts abbiegen usw.
- Input: RGB, Fahrzeugzustand, Navigationsbefehl
- Output: steering, throttle, brake
- 
Auswahl FRN als Feature/Image Encoder
- CIL als E2E-Modell
- FRN für visuelle Feature-Extraction
- CIL übernimmt die von FRN erzeugten Features und kombiniert mit Speed oder Steer
- FRN kann auch Event Daten extrahieren und den Output an CIL kombiniert übergeben
