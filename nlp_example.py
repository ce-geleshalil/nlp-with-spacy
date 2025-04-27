import spacy

# Türkçe modelini yükle
nlp = spacy.load("tr_core_news_lg")

# Örnek bir Türkçe metin
text = """
Türkiye'nin başkenti Ankara'dır. İstanbul, tarihi ve kültürel zenginlikleriyle ünlüdür.
Ahmet, 15 Nisan 2025'te İzmir'de bir konferansa katılacak. Halil kardeş, yüksek lisansına dikkat et.
"""

# Metni işle
doc = nlp(text)

# 1. Cümlelere ayırma
print("Cümleler:")
for sent in doc.sents:
    print(f"- {sent.text}")

# 2. Kelime türü etiketleme (POS Tagging)
print("\nKelime Türü Etiketleme:")
for token in doc:
    print(f"Kelime: {token.text:<15} POS: {token.pos_:<10} Açıklama: {spacy.explain(token.pos_)}")

# 3. İsim varlıklarını tanıma (NER)
print("\nİsim Varlıkları (NER):")
for ent in doc.ents:
    print(f"Varlık: {ent.text:<20} Tür: {ent.label_:<10} Açıklama: {spacy.explain(ent.label_)}")