### ISTRUZIONI

sei un esperto di intelligenza artificiale e un programmatore Python.
il tuo compito è aiutarmi nella creazione di un sistema RAG, in particolare di un Query Rewriting RAG,
quindi di un RAG dove la query dell'utente viene riscritta e ottimizzata dall'LLM.

### PIPELINE

Il sistema riceve un json con il prompt dell'utente. Il prompt viene preso dall'LLM e ottimizzato, per poi venire vettorizzato 
ed effettuare la query al Vector DB. Il sistema infine deve restituire solamente i documenti trovati con la ricerca semantica, senza
la risposta dell'LLM.
La pipeline deve essere questa:
Query -> Query Rewriting -> Vectorization -> Vector DB -> Retrieved Documents without LLM's response.

### ISTRUZIONI DI PROGRAMMAZIONE

Usa il codice già scritto all'interno del progetto come punto di partenza. Sistema il progetto senza stravolgerlo.

### OUTPUT
Array JSON con massimo 5 ricette nel seguente formato.
I campi assenti o nulli vanno omessi:

```json
[
  {
    "url": "https://ricette.giallozafferano.it/Tiramisu.html",
    "titolo": "Tiramisù",
    "categoria": "Dolci",
    "immagine": "https://www.giallozafferano.it/images/237-23742/Tiramisu_650x433_wm.jpg",
    "difficolta": "Facile",
    "tempi": {
      "preparazione": "40 min",
      "cottura": "6 min",
      "nota": "Nota"
    },
    "costo": "Medio",
    "dosi": "8 persone",
    "ingredienti": [
      {
        "nome": "Mascarpone",
        "quantita": "750 g"
      }
    ],
    "macronutrienti": {
      "Energia": { "valore": "543,3", "unita": "Kcal" },
      "Carboidrati": { "valore": "34,3", "unita": "g" },
      "di cui zuccheri": { "valore": "19,3", "unita": "g" },
      "Proteine": { "valore": "12,1", "unita": "g" },
      "Grassi": { "valore": "40,4", "unita": "g" },
      "di cui saturi": { "valore": "21,3", "unita": "g" },
      "Fibre": { "valore": "0,5", "unita": "g" },
      "Colesterolo": { "valore": "216,5", "unita": "mg" },
      "Sodio": { "valore": "522,7", "unita": "mg" }
    },
    "tag_dietetici": ["Vegetariano"]
  }
]
```