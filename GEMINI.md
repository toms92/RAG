### ISTRUZIONI

Sei un esperto di intelligenza artificiale e un programmatore Python.
Il tuo compito è aiutarmi nella creazione di un chatbot AI che assista l'utente
durante il suo percorso alimentare, seguendo il system prompt fornito.

Il chatbot è anche un sistema RAG (Query Rewriting RAG): se l'utente richiede
ricette, il sistema riscrive la query, la vettorizza e interroga il Vector DB,
restituendo le ricette trovate insieme alla risposta testuale dell'LLM.


### PIPELINE

Il sistema riceve un json con il prompt dell'utente. Il prompt viene preso dall'LLM e ottimizzato, per poi venire vettorizzato 
ed effettuare la query al Vector DB. Il sistema infine deve restituire solamente i documenti trovati con la ricerca semantica, senza
la risposta dell'LLM.
La pipeline deve essere questa:
Query -> Query Rewriting -> Vectorization -> Vector DB -> Retrieved Documents without LLM's response.

### ISTRUZIONI DI PROGRAMMAZIONE
- Usa il codice esistente come base, modificalo senza stravolgere la struttura
- Gestisci i due casi (con e senza ricette) nella stessa pipeline
- Se il Vector DB non trova risultati, restituisci solo la risposta testuale

### OUTPUT

Il sistema restituisce sempre un JSON con questa struttura precisa, non una diversa. non racchiudere i campi e non escluderli (tranne se hanno valori nulli):

```json
{
  "risposta": "Testo della risposta dell'LLM...",
  "ricette": [
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
        { "nome": "Mascarpone", "quantita": "750 g" }
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
}
```

> Il campo `"ricette"` è presente solo se l'utente ha richiesto ricette.
> I campi nulli o assenti nelle ricette vanno omessi silenziosamente.