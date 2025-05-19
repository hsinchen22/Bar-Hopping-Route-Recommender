<a id="readme-top"></a>

## Bar Hopping Route Recommender

<img src="images/banner.png" width="100%">

<div align="center">
Tired of the same old watering holes? Craving a night of diverse vibes and tantalizing tipples?

RunTini redefines nightlife by pairing curated bar experiences with a guided night run. 🍸🏃‍♂️

<b>🔥🔥🔥 Explore the repo and see how we’re turning bar-hopping into a true journey!</b>

<img src="https://img.shields.io/badge/Python-FFD43B?logo=python&logoColor=blue" height="20"/>
<img src="https://img.shields.io/badge/Numpy-777BB4?logo=numpy&logoColor=white" height="20"/>
<img src="https://img.shields.io/badge/Pandas-2C2D72?logo=pandas&logoColor=white" height="20"/>
<img src="https://img.shields.io/badge/Selenium-43B02A?logo=Selenium&logoColor=white" height="20"/>
<img src="https://img.shields.io/badge/Sqlite-003B57?logo=sqlite&logoColor=white" height="20"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" height="20"/>
<img src="https://img.shields.io/badge/scikit_learn-F7931E?logo=scikit-learn&logoColor=white" height="20"/>
<img src="https://img.shields.io/badge/-HuggingFace-FDEE21?logo=HuggingFace&logoColor=black" height="20"/>
</div>

### Table of Contents
<ul>
    <li><a href="#1"><strong>About RunTini - Your Night, Your Route</strong></a></li>
    <ul>
        <li><a href="#1-1">How It Works - The Magic Behind the Crawl</a></li>
        <li><a href="#1-2">Built With - The Secret Sauce</a></li>
    </ul>
    <li><a href="#2"><strong>Getting Started - How to Unleash the Fun</strong></a></li>
    <ul>
        <li><a href="#2-1">Prerequisites - What You'll Need in Your Toolkit</a></li>
        <li><a href="#2-2">Installation - Setting Up Your Adventure</a></li>
    </ul>
    <li><a href="#3"><strong>License - Cheers to Open Source</strong></a></li>
    <li><a href="#4"><strong>Contact - Holler At Us</strong></a></li>
</ul>

<a id="1"></a>
## 🍻 About RunTini - Your Night, Your Route

Tired of scrolling bar reviews, trying to plan the perfect night out? Meet RunTini – your nightlife wingman with a runner’s high. We mix curated bar vibes with a 3–5 mile night run, turning your night into a drink-fueled adventure. Whether you’re chasing moody whiskey dens, electric cocktail spots, or beer-soaked pubs, we map out five epic stops and the route to hit them all.

**Run. Sip. Repeat.** This isn’t just bar-hopping—it’s bar-running. Let’s make your night legendary.

<a id="1-1"></a>
### ✨ How It Works - The Magic Behind the Crawl

Our system orchestrates your perfect bar hop with a blend of advanced AI and geographical wizardry:

* <a>User Query Analysis</a> - Your Wish is Our Command:<br/>
You input your desired bar characteristics (e.g., "speakeasy vibe with craft cocktails"). This natural language query is the starting point of our search.

* <a>Contextual Review Summarization</a> - The Vibe Decoder: <br />
We leverage the powerful multimodal model to process and condense extensive user reviews and visual data from Google Maps for numerous bars. This provides nuanced insights into each venue's atmosphere, offerings, and overall experience, going beyond simple ratings.

* <a>Vector Embedding and Similarity Search</a> - Finding Your Tribe: <br />
The summarized review text for each bar is transformed into high-dimensional vector embeddings, capturing the semantic meaning of the bar descriptions to identify bars with the most semantically similar descriptions.

* <a>Embedding Adaptation</a> - The Hit Rate Hero:<br />
To bridge potential vocabulary gaps between user queries and bar reviews, we employ a linear adapter layer attached to the embedding model. This adapter learns a transformation matrix that fine-tunes the query embeddings, specifically improving the hit rate of relevant bars from 56% to **76%**. This is achieved by better aligning the semantic representation of user intent with the embedded bar descriptions, leading to a higher recall in the initial search.
<div align="center" style="margin-bottom: 20px;">
    <img src="images/anc_pos.png" width="70%">
</div>


* <a>Reranking with Cross-Encoder</a> - The Precision Pour:<br />
The initial set of candidate bars undergoes a reranking stage using a more computationally intensive cross-encoder model which directly compares the user query with each candidate bar's full review summary, predicting a relevance score. This step ensures that the top recommendations are not only semantically similar but also highly pertinent to the specific nuances of your request.

* <a>Hamiltonian Path Optimization</a> - The Route Master:<br />
With the top 5 bars selected, we model the bar locations as nodes in a graph and solve the Hamiltonian Path Problem. Our goal is to devise an efficient and enjoyable route that prioritizes a linear progression through different streets, minimizing backtracking and maximizing the exploration of new areas between your chosen bars.

* <a>Gradio Interface</a> - Your Night, Delivered: <br />
Finally, the curated list of 5 bars, along with the optimized bar-hopping route, is presented to you through a user-friendly web interface built with Gradio. This interface allows you to easily view bar details, the suggested route, and embark on your personalized drinking adventure.

<a id="1-2"></a>
### 🧩 Built With - The Secret Sauce

We've concocted RunTini with a potent blend of these amazing technologies:

| Technology                                  | Role                                                                      |
| :------------------------------------------ | :------------------------------------------------------------------------ |
| [![Gemma 3](https://img.shields.io/badge/google--gemma--3--4b--it-Intelligent%20Core-blue)](https://ai.google.dev/gemma) | The intelligent core for understanding bar vibes from user reviews and photos. |
| [![Granite Embedding](https://img.shields.io/badge/ibm--granite--embedding--125m--english-Text%20to%20Insights-brightgreen)](https://huggingface.co/ibm-granite/granite-embedding-125m-english) | Transforms summaries into searchable insights, the key to finding your perfect match. |
| [![GPT-4](https://img.shields.io/badge/openai--gpt--4-Vector%20Search%20Enhancer-yellow)](https://openai.com/gpt-4)                | Powers the generation of natural-sounding user queries, enhancing our search precision. |
| [![BGE Reranker](https://img.shields.io/badge/BAAI--bge--reranker--v2--m3-Recommendation%20Refiner-orange)](https://huggingface.co/BAAI/bge-reranker-v2-m3) | The final touch, ensuring only the top-notch recommendations make it to you.     |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="2"></a>
## 🚀 Getting Started - How to Unleash the Fun
Ready to let RunTini plan your next unforgettable night out? Here's how to get started:

<a id="2-1"></a>
### ⚙️ Prerequisites - What You'll Need in Your Toolkit
Before you can embark on your RunTini adventure, make sure you have the following installed:

<a id="2-2"></a>
### ⚡ Installation - Setting Up Your Adventure
Follow these steps to get RunTini up and running:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="3"></a>
## 📜 License - Cheers to Open Source
Distributed under the Unlicense License. See LICENSE.txt for more information. This means you're free to use, modify, and distribute RunTini as you see fit – no strings attached!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="4"></a>
## 💬 Contact - Holler At Us
Have questions, suggestions, or just want to share your favorite bar crawl story? Feel free to reach out!

<div align="center">
<a href="https://github.com/hsinchen22">
  <img src="https://github.com/hsinchen22.png" width="80" style="border-radius: 50%; margin: 0 5px;"/>
</a>
<a href="https://github.com/??">
  <img src="https://github.com/??.png" width="380" style="border-radius: 50%; margin: 0 5px;"/>
</a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>