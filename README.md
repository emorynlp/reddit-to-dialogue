<h1> Reddit-to-Dialogue </h1>

<h3> Purpose </h3>

<b>Reddit-to-Dialogue</b> is a tool that *transforms* a Reddit post & its comments into a dialogue.

#

<h3> Developement </h3>

This project began at *Emory University* in the [Emory NLP Lab](https://www.emorynlp.org/) under the direction of [Dr. Jinho Choi](https://www.emorynlp.org/faculty/jinho-choi).

The development constituted two separate undergraduate honors theses, undertaken by [Daniil Huryn](https://www.emorynlp.org/bachelors/daniil-huryn) and [Mack Hutsell](https://www.emorynlp.org/bachelors/mack-hutsell), and resulted in a long paper accepted to [COLING 2022](https://coling2022.org/).

#

<h3> Installation </h3>
!! Pip Package Coming Soon !!
`pip3 install reddit-to-dialogue`

#

<h3> Usage </h3>

<h4> Input </h4>

Data should be in a JSON format, organized as defined by [PRAW](https://praw.readthedocs.io/en/stable/). Example in reddit folder.

<h4> Output </h4>

Dialogues will be returned in the format (example in exampleoutput):

```
[{
    "sid": "",
    "link": "",
    "title:": "",
    "text": "",
    "author": "",
    "created": unix timestamp,
    "updated": unix timestamp,
    "over_18": boolean,
    "upvotes": integer,
    "upvote_ratio": decimal value 0 - 1,
    "response": [
        "",
    ],
    "dialogue": [
        "",
    ],
    "score":
}, ]
```

Where *response* is a list of Speaker 2 statements and *dialogue* alternates Speaker 1 and Speaker 2 statements.
