import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Charger les modèles une fois (cache)
@st.cache_resource
def load_models():
    # 1. Modèle de correction grammaticale (T5)
    correcteur = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
    
    # 2. Modèle de génération d'images (Stable Diffusion)
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # GPU si disponible
    
    return correcteur, pipe

correcteur, pipe = load_models()

# Interface Streamlit
st.title("📝 IA Journaliste : Correcteur & Générateur d'Images")
st.markdown("""
Corrigez vos articles et générez des illustrations automatiquement !
""")

# Zone de texte pour l'article
article = st.text_area("Collez votre article ici :", height=300)

if st.button("Corriger et Générer l'Image"):
    if not article:
        st.error("Veuillez entrer un texte.")
    else:
        # 1. Correction Grammaticale
        with st.spinner("Correction en cours..."):
            texte_corrige = correcteur(article, max_length=1024)[0]['generated_text']
        
        st.subheader("✅ Article Corrigé")
        st.write(texte_corrige)

        # 2. Génération d'Image (limité à 500 caractères pour éviter les erreurs)
        with st.spinner("Création de l'illustration..."):
            try:
                image = pipe(texte_corrige[:500]).images[0]
                st.subheader("🖼️ Illustration suggérée")
                st.image(image, caption="Généré avec Stable Diffusion")
            except Exception as e:
                st.error(f"Erreur lors de la génération d'image : {e}")