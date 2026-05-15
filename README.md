Belgium Street Art Explorer

The dataset comes from Street Art Cities, a community platform that documents street art worldwide. I used the dataset they provided of street artworks collected across Belgium, containing 7,199 pieces of work. Each record contains the title, artist names, location of the work, and a short description.

The model I used is BLIP (Bootstrapped Language-Image Pre-training), an image captioning model developed by Salesforce Research and available on Hugging Face. BLIP is a vision-language model trained on large-scale image-text pairs that can generate natural language descriptions of images. In this application, it takes an image uploaded by the user, generates a text caption describing what it sees, then matches that caption against related artworks from the dataset and returns the results back to the user.

This application contains three main sections. The Map allows users to locate artworks by scrolling through the cities, giving a stronger sense of geographic context. The Archive stores all the data from the dataset with clear structured information for each work, and allows users to filter by artist, location, or style. The Search tab allows users to upload their own street art photos and find works from the dataset they might be interested in.

The hardest part for me was building the application through Streamlit. I have previous experience with CSS and HTML, so the styling was actually the easier part — but building the underlying structure of the app and understanding how Streamlit handles state, layout, and components was the steepest learning curve. Getting the data, model, and interface to work together as one cohesive thing took the most time and iteration.


Data Source
Street Art Cities https://streetartcities.com

Dataset: Belgian artworks export, May 2026

Model
BLIP Image Captioning (base) — Salesforce Research
https://huggingface.co/Salesforce/blip-image-captioning-base