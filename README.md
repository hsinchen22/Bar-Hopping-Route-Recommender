<a id="readme-top"></a>
# Bar Hopping Route Recommender
<div>
    <img src="images/banner.png" width="100%">
    <br />
    <p align="center">
    Tired of the same old watering holes? Craving a night of diverse vibes and tantalizing tipples?
    <br /><strong>
    RunTini redefines nightlife by pairing curated bar experiences with a guided night run. </strong>🍸🏃‍♂️
    <br />
    <a href="https://github.com"><strong>Explore the repo and see how we’re turning bar-hopping into a true journey »</strong></a>
    <br />
    <br /> 
    <p align="center">
    <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" height="20"/>
    <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" height="20"/>
    <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" height="20"/>
    <img src="https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=Selenium&logoColor=white" height="20"/>
    <img src="https://img.shields.io/badge/Sqlite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" height="20"/>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" height="20"/>
    <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" height="20"/>
    <img src="https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black" height="20"/>
    </p>
</div>

### Table of Contents
* **🍻 About RunTini - Your Night, Your Route**
    * 🍿 [Demo Video - Watch it in Action!](#1-1)
    * ✨ [How It Works - The Magic Behind the Crawl](#1-2)
    * 🧩 [Built With - The Secret Sauce](#1-3)

* **🚀 Getting Started - How to Unleash the Fun**
    * ⚙️ [Prerequisites - What You'll Need in Your Toolkit](#2-1)
    * ⚡ [Installation - Setting Up Your Adventure](#2-2)

* **🗺️ Roadmap - Future Cocktails**
    * 🌱 [Short-Term Goals](#3-1)
    * 🔮 [Long-Term Vision](#3-2)

* **📜 License - Cheers to Open Source**

* **💬 Contact - Holler At Us**

## 🍻 About RunTini - Your Night, Your Route

Tired of scrolling bar reviews, trying to plan the perfect night out? Meet RunTini – your nightlife wingman with a runner’s high. We mix curated bar vibes with a 3–5 mile night run, turning your night into a drink-fueled adventure. Whether you’re chasing moody whiskey dens, electric cocktail spots, or beer-soaked pubs, we map out five epic stops and the route to hit them all.

<strong>Run. Sip. Repeat.</strong>
<br/>
This isn’t just bar-hopping—it’s bar-running. Let’s make your night legendary (and slightly sweaty).

<a id="1-1"></a>
### 🍿 Demo Video - Watch it in Action!

<a id="1-2"></a>
### ✨ How It Works - The Magic Behind the Crawl

Our system orchestrates your perfect bar hop with a blend of advanced AI and geographical wizardry:

* **<span style="font-size: 1.2em;">❓</span> User Query Analysis - Your Wish is Our Command:** <br/>
You input your desired bar characteristics (e.g., "speakeasy vibe with craft cocktails"). This natural language query is the starting point of our search.

* **<span style="font-size: 1.2em;">🧠</span> Gemma 3 - Contextual Review Summarization:** <br />
We leverage the powerful multimodal model to process and condense extensive user reviews and visual data from Google Maps for numerous bars. This provides nuanced insights into each venue's atmosphere, offerings, and overall experience, going beyond simple ratings.

* **<span style="font-size: 1.2em;">🧬</span> Vector Embedding and Similarity Search - Finding Your Tribe:** <br />
The summarized review text for each bar is transformed into high-dimensional vector embeddings, capturing the semantic meaning of the bar descriptions to identify bars with the most semantically similar descriptions.

* **<span style="font-size: 1.2em;">🔑</span> Embedding Adaptation - The Hit Rate Hero:**<br />
To bridge potential vocabulary gaps between user queries and bar reviews, we employ a linear adapter layer attached to the embedding model. This adapter learns a transformation matrix that fine-tunes the query embeddings, specifically improving the hit rate of relevant bars <font color="#17C3FF">from 56% to 76%</font>. This is achieved by better aligning the semantic representation of user intent with the embedded bar descriptions, leading to a higher recall in the initial search.
<div align="center" style="margin-bottom: 20px;">
    <img src="images/anc_pos.png" width="70%">
</div>

* **<span style="font-size: 1.2em;">📊</span> Reranking with Cross-Encoder - The Precision Pour:**<br />
The initial set of candidate bars undergoes a reranking stage using a more computationally intensive cross-encoder model which directly compares the user query with each candidate bar's full review summary, predicting a relevance score. This step ensures that the top recommendations are not only semantically similar but also highly pertinent to the specific nuances of your request.

* **<span style="font-size: 1.2em;">🚶</span> Hamiltonian Path Optimization - The Route Master:**<br />
With the top 5 bars selected, we model the bar locations as nodes in a graph and solve the Hamiltonian Path Problem. Our goal is to devise an efficient and enjoyable route that prioritizes a linear progression through different streets, minimizing backtracking and maximizing the exploration of new areas between your chosen bars.

* **<span style="font-size: 1.2em;">🚚</span> Gradio Interface - Your Night, Delivered:** <br />
Finally, the curated list of 5 bars, along with the optimized bar-hopping route, is presented to you through a user-friendly web interface built with Gradio. This interface allows you to easily view bar details, the suggested route, and embark on your personalized drinking adventure.

<a id="1-3"></a>
### 🧩 Built With - The Secret Sauce

We've concocted RunTini with a potent blend of these amazing technologies:

| Technology                                  | Role                                                                      |
| :------------------------------------------ | :------------------------------------------------------------------------ |
| [![Gemma 3](https://img.shields.io/badge/google--gemma--3--4b--it-Intelligent%20Core-blue)](https://ai.google.dev/gemma) | The intelligent core for understanding bar vibes from user reviews and photos. |
| [![Granite Embedding](https://img.shields.io/badge/ibm--granite--embedding--125m--english-Text%20to%20Insights-brightgreen)](https://huggingface.co/ibm-granite/granite-embedding-125m-english) | Transforms summaries into searchable insights, the key to finding your perfect match. |
| [![GPT-4](https://img.shields.io/badge/openai--gpt--4-Vector%20Search%20Enhancer-yellow)](https://openai.com/gpt-4)                | Powers the generation of natural-sounding user queries, enhancing our search precision. |
| [![BGE Reranker](https://img.shields.io/badge/BAAI--bge--reranker--v2--m3-Recommendation%20Refiner-orange)](https://huggingface.co/BAAI/bge-reranker-v2-m3) | The final touch, ensuring only the top-notch recommendations make it to you.     |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🚀 Getting Started - How to Unleash the Fun
Ready to let RunTini plan your next unforgettable night out? Here's how to get started:

<a id="2-1"></a>
### ⚙️ Prerequisites - What You'll Need in Your Toolkit
Before you can embark on your RunTini adventure, make sure you have the following installed:

<a id="2-2"></a>
### ⚡ Installation - Setting Up Your Adventure
Follow these steps to get RunTini up and running:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🗺️ Roadmap - Future Cocktails
We're always looking to enhance your bar-hopping experience! Here are some exciting features we're planning for the future:

<a id="3-1"></a>
### 🌱 Short-Term Goals

* **Real-time Bar Information:** <br />
Integrate real-time data on bar crowd levels and potential wait times to help users make more informed decisions about which bars to visit and optimize their bar-hopping experience in real-time.

* **Mobile Apps & Community Route Sharing:** <br />
Build native mobile apps that integrate RunTini's core functionality with features for users to create, share, and discover bar-hopping routes within the community.

<a id="3-2"></a>
### 🔮 Long-Term Vision

* **Feedback-Driven Personalization:** <br />
Integrate feedback mechanisms (e.g., thumbs up/down on recommendations, saving routes) to build user profiles and improve future route suggestions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📜 License - Cheers to Open Source
Distributed under the Unlicense License. See LICENSE.txt for more information. This means you're free to use, modify, and distribute RunTini as you see fit – no strings attached!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 💬 Contact - Holler At Us
Have questions, suggestions, or just want to share your favorite bar crawl story? Feel free to reach out!

✈️ **Hsin Chen** - hsinchen@stanford.edu
<br />
☕ **Justin 自己寫** -
<br />

**Project Link**: https://github.com/hsinchen22/RunTini

<p align="right">(<a href="#readme-top">back to top</a>)</p>
