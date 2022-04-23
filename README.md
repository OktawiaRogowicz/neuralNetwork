<h2 align="center">
  
  🧠 <code>QUICK, HELP!</code> 🧠
  
  Web application categorizing hand drawn pictures and shapes using neural network
  
  🖼️
  
  <code>back-end</code> </h2>

<div align="center">
This is back-end respository for my thesis. <a href="https://github.com/OktawiaRogowicz/neural-network-front">Front-end can be viewed here</a>! 

The aim of the project as a whole is an implementation of a website allowing its users to play cha-
  rades with a neural network, similarly to Google’s <b>Quick, Draw!</b>. 
  
.

<strong>Heroku</strong>: <a href="neural-network-react.herokuapp.com"><strong>LIVE SITE</strong></a>
</div>

<h1><code>Overview</code></h1>

> Front-end part of the project has been designed using JavaScript and its library React. It
allows not only to game, but also to collect all drawn pictures, categorizing each one of
them accordingly. Collected data is saved in a data base created in Cloudinary. During
the development process, about 1000 unique images drawn by individual users has been
collected, which then have been subjected to data augmentation. The neural network’s
accuracy oscillates around 81%.

> After starting the game, the exchange with neural network starts. During ten rounds,
the player has the task to draw ten drawing prompts, which neural network then tries to
categorize correctly. The user then can look into its results and errors, and eventually -
play again.



<div align="center">
  <img src="https://github.com/OktawiaRogowicz/ip-address-tracker/blob/main/src/ip-address-tracker-master/img.png"
    alt="Screenshot" width="500"/>
</div>



<div align="center">
  HTML <strong>||</strong> CSS <strong>||</strong> flexbox <strong>||</strong> React <strong>||</strong> Styled components
  
  IP geolocation API <strong>||</strong> Leaflet
  
<strong>Heroku</strong>: <a href="neural-network-react.herokuapp.com"><strong>LIVE SITE</strong></a>
</div>


To make my project possible, I needed to divide it into three parts.

<code>first part</code>

As one of the premises was training my own neural network, I needed big enough database, as anything less than one hundred images per category will not be useful. To collect my own database, I decided to create a website, on which players will be able to help me collect pictures for the neural network, <b>second part</b> of the project - thus the name of the project, back from the times when I promoted it online.

<code>second part</code>
...was creating neural network. After deciding on the architecture, I wrote few scripts - that can be viewed in the root folder - preprocessing collected images into much more accessible contect. Afterwards, using TensorFlow and Keras, I trained a neural network with a success rate oscillating around 81%. 

<code>third part</code>
Putting together neural network and a website prepared before. Using TensorFlow.js and Node.js, I let the user to exchange data with my neural network - and finished the game. Afterwards, thanks to JavaScript, I was able to put a few more details into the page.

### Summary

During building the website, about one hundred people had interacted with it. As a result, the latest neural network version has been taught using about 1200 images unique to the thesis.

Even though it is an impressive number, it is not enough for the size of this project, or rather – the specifics of its neural network. Even though its accuracy on test set has achieved around 81.5 percent, in a real-life setting, some slight imperfections made by users are enough to categorize the image wrong. 

The easiest mistake and its origin to spot is “sun” categorization. It is the only one with few evident, sharp lines coming out of it. Every time the player makes a mistake and draws a sharp edge, line, or angle, it is recognized as a ray of sun. It also seems to be the category neural network defaults to, since it has a circular shape, just like a “cookie”, a “moon”,  and a part of a “mug”, but also some sharp and chaotic lines like “grass” or “broccoli”. It matches a lot of drawings that players have not intended to be a “sun” in the slightest. 

### What I learned

Reasons to do this project were pretty simple - it is something I wanted to learn and try out, so I did. As it turns out, the knowledge is not only practical and beneficial, but the process of learning, for me, was also simply enjoyable and amusing. I found JavaScript and React easy to use, with great possibilities, and neural networks fascinating – observation of expanding database and how the network's thought process alternates with each change was absorbing and informative.

### Continued development

In the original Google's <b>Quick, Draw!</b> game images are predicted in real-time. Making it work with the current database would be problematic; however, rebuilding the code accordingly and collecting new data would be possible. In such a case, real-time predictions could happen and using recursive neural networks to guess how player's next line could be drawn.

<h1><code>Author</code></h1>

- Website - [Add your name here](https://www.your-site.com)
- Frontend Mentor - [@yourusername](https://www.frontendmentor.io/profile/yourusername)
