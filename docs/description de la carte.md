<aside>

**Objectif de la carte :** monter un pipeline qui va recouper toutes les exigences de tous les référentiels et identifier les correspondances qui peuvent exister.

Le but c’est que si l’exigence A du ref X est similaire à l’exigence B du ref Y ; si un user est conforme à A il sera automatiquement conforme à B.

---

Logique (théoriquement faisable, mais pratiquement impossible) :

Comparer toutes les exigences entre elles → 70 référentiels * 200 exigences chacun = 14 000 exigences

N_total de couples à comparer ~200 millions (avec 99.9% de bruits = couts et temps démesurés pour rien)

---

1. Normalisation des exigences : requirementNormalized, id, framework, chapter, title, description, tags
2. Vectorisation des exigences
3. Filtrage des couples plausibles : on élimine 99.9% de comparaisons inutiles
4. Analyse “taux de chevauchement” IA

Les similarités entre exigences c’est pas seulement : “est-ce que A ressemble à B ?"

mais plutôt : "si A est conforme, est-ce que ça couvre B ?"

Le LLM retourne un scoring entre 0 et 1 de : coverage_A_to_B ; coverage_B_to_A ; confidence ; et on obtient une relation de couverture :

- d'équivalence
- A couvre B
- B couvre A
- Partielle
- Pas de lien

Ca permettra:

- d’identifier automatiquement les équivalences entre référentiels
- de propager la conformité d’un référentiel vers un autre
- de réduire de 99.9% le travail d’audit
- de générer automatiquement des matrices de correspondance multi-référentiels

On passe de 200 millions de comparaisons à quelques dizaines/centaines de milliers de comparaisons utiles : faisable en pratique

**monter le pipeline sur CIS et NIST d’abord**

</aside>

- Vectoriser les question, mise en cache directement sur la machine (RAG)
- Regrouper avec un sytème de scoring
- Recoupage par recherche sémantique
- Le 1% restant le mapper
- La complexité  : dans le meilleur des cas, tel exigence regroupe tel autre exigence
- Trouver un  bon compromis