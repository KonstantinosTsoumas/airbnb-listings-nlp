<br />
<div align="left">
  <a href="https://github.com/KonstantinosTsoumas/airbnb-listings-nlp">
  </a>

  <h3 align="Center">An Empirical Test of Aristotle's Persuasion with Airbnb Listings<h3>
    
  <h3 align="Left">This project tries to answer questions such as which is the most appropriate content for houses that are more (less) premium and does the type of promotion matter. Latent Dirichlet Allocation, Sentiment Analysis (sentence, word based) and Tobit regression were used.
   If this repo was useful in any way, please don't forget to cite, leave a star, share the fun.<h3>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#dataset">Dataset</a>
      <ul>
        <li><a href="#Dataset">Dataset</a></li>
        <li><a href="#Evaluation Metrics">Evaluation Metrics</a></li>
      </ul>
    </li>
    <li><a href="#Listings">Listings</a></li>
    <li><a href="#Persuasion Theories">Persuasion Theories</a></li>
    <ul>
        <li><a href="#Ethos">Ethos</a></li>
        <li><a href="#Logos">Logos</a></li>
        <li><a href="#Pathos">Pathos</a></li>
      </ul>
    </li>
    <li><a href="#Useful Links">Useful Links</a></li>
    <li><a href="#Contribute">Contribute</a></li>
    <li><a href="#contact">contact</a></li>
  </ol>
</details>

<!--Abstract-->
## Abstract

  Although the demand for digital accommodation platforms is increasing like never before, it is yet not known what the most appropriate promotion content for specific product types is. This research contributes to the previous literature by investigating how hosts, on the Airbnb platform, can persuade the decision of potential guests to book their listings. 
  This was implemented by matching Aristoteles's Rhetoric theory, Logos (Logical Proof), Ethos (Credibility), and Pathos (Emotion) with certain variables, and by manually calibrating the occupancy rate as a performance measurement. From the aforementioned, only the first persuasive mode is matched with utilitarian products, while the remaining were related to hedonic products. 
  Latent Dirichlet Allocation (LDA) is applied to build an understanding of the topics used in the dataset, and then Sentiment Analysis is implemented to capture the valence of hosts when writing and describing themselves and their listings. The final model is estimated using Tobit regression. Persuasion through Logos (Logical proof) has the smallest negative effect on the occupancy rate. Likewise, Pathos (Emotion) also has a negative effect on the occupancy in comparison to Ethos (Credibility), which positively affects the occupancy rate suggesting a positive effect also on estimated bookings. I provide several suggestions for further research based on these results.
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--About the project-->
### Dataset

  The initial dataset was provided from a publicly available website that contains relevant Airbnb data for specific dates throughout the year in various continents, countries. To extract the data, open-source technologies and maps were used in order for the data to be more visual. The final dataset, which was used for analysis, is a merged dataset of detailed listings and reviews providing a unique set of data. 

https://github.com/KonstantinosTsoumas/airbnb-listings-nlp/blob/main/imgs/listings.png
    
https://github.com/KonstantinosTsoumas/airbnb-listings-nlp/blob/main/imgs/listings_zoomed.png
### Evaluation Metrics
  In order to be able to measure the performance across listings (hedonic and utilitarian), the occupancy rate was used. This measurement however, is not publicly available and had to be calculated manually based on the "San Fransisco Model". In addition, regarding Tobit regression both Pseudo-R squared and Variance Inflation Factor have been taken into account. 
  What's more, the initial dataset appears to have an upper bound to the number of words used in the text which sometimes lead to an unexpected cut of the sentence. 
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--Listings-->
## Listings

<img src="https://github.com/KonstantinosTsoumas/airbnb-listings-nlp/blob/main/imgs/listings.png" width="700" height="500">
<img src="https://github.com/KonstantinosTsoumas/airbnb-listings-nlp/blob/main/imgs/listings_zoomed.png" width="700" height="500">

  The figures above demonostrate the spread of the Airbnb listings throughout the city of Amsterdam. The second figure zooms into the center of Amsterdam to provide more clear insights about the given listings. Both of these figures are presented as spatial data (after transforming latitude and longitude) and presented to the common standard World Geodetic System (WGS84). Unsuprisingly, it is obvious that many listings are gathered around the center area of Amsterdam, while a noticeable number of listings can be found on the West side of Amsterdam in comparison to the East side where not many listings are available. 
  
  An interesting plot is the following, price against room type. The figure, as the xaes and the legend display, illustrate the distribution of the prices acrosss the types of available rooms in the dataset. It can be inferred from the graph that the most expensive type of accommodation is a private room with an average price of almost $100 per night. The x axe is limited between (0, 500) for visualisation purposes.
  
<img src="https://github.com/KonstantinosTsoumas/airbnb-listings-nlp/blob/main/imgs/price_room_type.jpeg" width="600" height="400">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--Persuasion Theories-->
## Persuasion Theories

## Ethos
To pursue customers through Ethos in the Airbnb setting, the strengthening of the host’s credibility needs to be addressed.  In Aristotle’s rhetoric theory, ethos refers to the character (credibility) of the speaker and specifically when the speech is manipulated in a way of presenting the speaker's worth or credence. In this project, we used four indicators of the host credibility: a super host badge, the number of reviews & ratings, and communication, ID verification. 

## Logos
Concerning persuasion through logos, can be accomplished by presenting arguments in order for the speaker to make his statement persuasive through logical proof. Persuasion is paramount in the Airbnb environment for the host-guest (stranger-stranger) communication and it serves as a primary element of the relationship between the owner and the potential customer. In this project, Logos has been calculated based on: safety features, amenities, average review score of listing and if the host has a profile picture.

## Pathos
According to Aristotle, persuasion through pathos comes when the audience is led by a speech from which people can feel emotion or passion, and this has an impact on the judgment call they are going to make later. In this project, emotional words were captured by using Sentiment Analysis. Specifically, the emotional words found in the "Description" and "Host about" columns were measured to get the valence of hosts. It must be highlited that both negative and positive sentiments are considered emotional. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--Useful Links-->
## Useful Links

1. [What is NLP?](https://www.sciencedirect.com/topics/computer-science/natural-language-processing)
2. [A short introduction to LDA](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2)
3. [A short introduction to Sentiment Analysis](https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17)
4. [What is Airbnb?](https://www.airbnb.com/help/article/2503)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--Contribute-->
## Contribute

- Clone [this](https://github.com/KonstantinosTsoumas/airbnb-listings-nlp) repository:

```bash
git clone https://github.com/KonstantinosTsoumas/airbnb-listings-nlp
```
- Contribute by making changes [PR](https://github.com/KonstantinosTsoumas/airbnb-listings-nlp/pulls).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Konstantinos Tsoumas - [LinkedIn](https://www.linkedin.com/in/konstantinostsoumas/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
