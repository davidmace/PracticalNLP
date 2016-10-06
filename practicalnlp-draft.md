

#Table of Contents
1. Infrastructure and Framework
  a. NLP System Design Philosophy
  b. Data Scale Intuition
  c. Data You Expect vs Data You Get
  d. Machine Learning Framework
  e. Basic Word Models (Tfidf)
  f. Unigram + Bigram Tricks
    1. PMI trick to lessen features
  g. Compact Models (LDA + WMD + x2vec)
  h. Auditing Model Performance
  i. Auto-Updating Models
  j. Notebook Dev Cycle
2. Specific Components
  1. Named Entities (person names, locations)
  2. Regex Entities (email, url, phone, number, number range, )
  3. Time
  4. Spelling Correction
  5. Colloquialisms
  6. Parsing
  7. Dependency Parse Tricks
  8. Wordnet Annotations
  9. Tracking Bot State
3. Integrating Deep Learning
  1. Challenges of Deep Learning
  2. Adding Memnets to the Machine Learning Framework
  3. Systems that Improve as DL Research Improves
4. Flows
  1. Bot flow
  2. NLP Recommender flow
  3. Twitter classification flow




###Resources

Dictionaries:
> <i class="icon-download"></i> [common-non-entity-words.txt](http://www.google.com)

Unigrams, Bigrams, Trigrams:
> <i class="icon-download"></i> [common-non-entity-words.txt](http://www.google.com)


###Named Entity Extraction
Example of extracting names and locations from text.

  1. hi im sakthi -> {sakthi:name}
  2. minh is in south dakota -> {minh:name, south_dakota:location}
  3. san jose is better than sf -> {san_jose:location, sf:location}

The simplest approach to recognizing names is to make a big list of names from census records, Wikipedia, etc. This is possible, but it’s more scalable to use an established Named Entity Recognizer because we can expect its performance to improve over the next few years. It’s also hard to get name lists working well for foreign names (ie. John-Paul, Veeral, Vi)—there are so many rare possibilities that you’ll probably miss the last 10%.

Stanford’s NER (Named Entity Recognizer) scales poorly (anecdotal observation, not benchmarked analysis). I recommend Google’s NL API Entity Recognizer because it's fast, scalable, and decently although not perfectly accurate.

NERs are notoriously bad at recognizing lowercase text (ie. sarthak went to bangalore), which is unfortunate because users rarely take the time to capitalize. There’s a hack to get around this. The word list below is my semi-manually curated list of common non-entity words. If you capitalize every word that isn’t in this list, it will improve the odds that the Google NER recognizes rarer entities with only a small boost in false positives.

  i went park in delhi yesterday -> i went park in Delhi yesterday

Look at the section on Word Tokenization to see how to separate words.

Common Word List:
> <i class="icon-download"></i> [common-non-entity-words.txt](http://www.google.com)

Google NER Docs:
https://cloud.google.com/natural-language/docs/


###Typed Entity Extraction

> <i class="icon-fire"></i> **Important principle:** 
> It might seem like a lot of the cases we deal with in this chapter are very esoteric so you shouldn’t need to worry about them. Because human language is very long-tailed (we get really weird, unexpected input), not accounting for these seemingly rare cases will lead to bad performance because there are so many weird cases.

 

> <i class="icon-fire"></i> **Important principle:** 
> These entity extraction tasks are logically simple and run quickly. Rather than run them as microservices, it makes sense to just include them as functions in your main algorithm flow code.

#### Emails

Email extraction is easy because the format <kbd>x@y.z</kbd> rarely if ever picks up false positives.

Here's the regex:
```
\b[\w\.\-!#$%&'*\+\/=\?\^_`{\|}~]+@[\w\.\-]+\.[\w\-]+\b
```

And some simple test cases:

```
a924-g@gmail.com, true
david-c.mace@hot-mail.com, true
send @ 5pm.tomorrow, false

```
And some ready-to-deploy code (NOTE: do i need this):
[email-regex.py](github.com)
[email-regex.java](github.com)

> <i class="icon-info"></i>**Extra Note:** 
> If internationalization is a big concern, you should account for unicode characters (ie. david本@abc.com). The regex above works if you first convert your string to [Punycode](https://en.wikipedia.org/wiki/Punycode).

####URLs

URL extraction is more complicated because all of the following formats need to be recognized:
```
https://www.google.com
www.google.com
google.xyz
google.xyz/dogs-cats?dogs=500
david.google.xyz:3000/dogs#33
google.xyz?i+like+turtles
```
  
And the following formats need to be ignored :
```
bye.done with trial
I am at Google.come to the park.
user@google.com
```

The best regex I’ve seen looks for the pattern <kbd>wx.yz</kbd> where w is http(s):// and/or www , x contains valid characters for a domain name, y is in a list of valid domain endings, and z belongs to a set of valid characters for url parameters, port, etc.

Here's that regex (should run as case-insensitive):
```
\b(?>https?:\/\/)?[\w\-\.]+\.(?>
+'|'.join(top_level_domains)+
)[\w-._~:\/\?#\[\]\@\!\$&%'\(\)\*\+,;=]*\b
```

And here's my list of common url endings that has 98% coverage:
> <i class="icon-download"></i> [top-level-domains.txt](https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains)

The easiest way to prevent matching emails as urls is to first extract emails then make sure your urls are not substrings of those emails.

> <i class="icon-info"></i>**Extra:** 
> I think it's overkill but if you need better than 98% coverage, you can find the updated full list here: [wikipedia top level domains](https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains)

> <i class="icon-info"></i>**Extra:** 
> Like emails, urls can contain unicode characters. For instance .рф is the top level domain for ~0.1% of websites. If internationalization is a big concern, the regex above works if you first convert your string to [Punycode](https://en.wikipedia.org/wiki/Punycode).

####Phone Numbers
Phone number extraction is more complex than you might imagine. If we want to work on international numbers, we have to handle all of these cases:
```
+12-555-555-5555
555 555 5555
5555555
555.5555
555-5555 x.7
555-5555, Ext 8
555-5555 extension 3
(01 55) 5555 5555
0455/55.55.55
05555 555555-55
```

But not these:
```
4500
I have 100. 1000 are on the way
I have between 100-1000 berries
```
Without taking nearby words into account, it’s not possible to differentiate some phone numbers from normal numbers (ie. 5555555 or 100-1000). This is decently rare though so I usually just mark any number with seven or more digits/dashes as a phone number.

Here's the regex (should run as case-insensitive):
```
[\s\-\d\+\(\)\/\.]{7,}[\s\.,]*(?:x|ext|extension)?[\s\.,\d]*\d
```

###Numbers

Extracting digit-based numbers is simple. Unfortunately people often don't write numbers as digits. Regex isn't the best move since we probably want to map 72 thousand -> 72000 rather than just recognizing that 72 thousand is a number.

Here are some examples of the weird cases we need to handle:

```
9000
9,000
9.0
-9
1.5m
a million
four hundred thousand and twenty eight
72 thousand
one and a half
```

Also, we should ignore numbers that we have previously identified as parts of phone numbers.

INSERT CODE


###Time Extraction

Times are the hardest entity to extract properly. Ideally we also want to map each time string to a numerical calendar value which we can query later. Here are some examples to show why it's so hard:

```
yesterday at 7:30am
second Wednesday of April
in 35 minutes
January 4th at 3pm
tomorrow at half past noon
```

Wit.AI’s Duckling has the highest accuracy of any library I've seen. It supports English, Spanish, French, Italian, and Chinese, which together cover 67% of online spoken language. More importantly it's written in a way (probabilistic models) that makes it easily extensible to further languages in the future.

Duckling additionally offers number, email, url, etc parsing; however, I’ve found regex parsers to work better on all entities except time (as of Sept 2016). 

Duckling is written in clojure so I wrapped it in a simple server because A. that's the easiest way for me to call it from my main python code and B. more importantly, Duckling does a non-negligible amount of work so I'll want to easily scale it up/down later (which is easiest if it's a separate service that I can move to a separate server from my main logic). 

> <i class="icon-download"></i> **Duckling Server Code:** https://github.com


###Spelling Correction

Hunspell is a high accuracy, easy to deploy spellchecking solution. It's deployed in Chrome and Safari among others, which means that it's likely to continue being supported at least in the near term. It also supports a variety of languages, which is important for internationalization.

The only nuance to the code is that out of the box, hunspell doesn't recognize certain entity types like ordinals (4th) and people names, so we need to ignore those suggestions. This means that entity recognition has to happen before spelling correction. Thankfully Google's NL API Named Entity Recognizer does some basic spelling correction on entities, which helps but isn't perfect. We can expect Named Entity Recognizers to get better at this over time.

Below is a microservice that wraps the hunspell open source library. Hunspell does a considerable amount of work, which is why it makes sense to deploy it as a separate service.

> <i class="icon-download"></i> **Hunspell Server Code:** https://github.com


###Internet Slang Correction

Spelling correctors don't pick up on Internet slang <kbd>thx->thanks</kbd> <kbd>kk->okay</kbd>. For example we might want to respond "No problem" to any form of "thanks" "thks" "thx" "ty". If we have all the data in the world, an algorithm should be able to figure out each of these responses separately, but in most cases we will never see at least one of these forms for "thanks" in the training data.

Fixing colloquialisms will only provide negligible improvement if you have a large amount of data or features (ie. unigram model for Tweet classification) since the few additional features needed to detect multiple forms of a word are a small percentage of the total model features.

Look at the section of Word Tokenization to see how to separate the words in a sentence.

Here is my list of internet abbreviations and their full spellings:
> <i class="icon-download"></i> [internet-abbreviations.csv](github.com)

.
> <i class="icon-info"></i>**Extra:** 
I only extracted internet abbreviations for English. If you would like to generate an internet abbreviation list for other languages, here's a process that works. Get a large amount (1m+ lines) of online conversation data (I used a day's worth of public Tweets). Eliminate every word that is present in a dictionary of known words (I included a link in Resources at the start of Section 3). Sort the remaining words by frequency. Manually go through the highest frequency 500 words and extract all of the Internet abbreviations in the list.

###Guessing Ethnicity, Gender, Age from Name


###Parsing (TODO include POS)

A dependency parse is a structured representation of a phrase or sentence. It tells us the relationships between words. For example the following sentence contains each of the dependencies below.
> I gave the dogs to Mary ->
> (root,root,gave)
> (nsubj,gave,I)
> ...

To see why this is useful, consider the example below. Imagine we want to know who the dogs were given to.

> I gave the dogs quickly to Lily
> I had given the dogs yesterday to Bill
> I will give the dogs from the park to Bob

The language is too complex to easily extract the relationship between the verb "give" and the people. However, the dependency parse has given us structured information so all we have to do is look for dependencies of the form <kbd>(iobj,give,ANSWER)</kbd>.

There are a few options for parsing. Stanford Parser is historically the most accurate but it did not scale well for my past projects and I wasted time trying to optimize its internal resources. Hosting your own Tensorflow-based Parsey Mcparseface as a microservice seems like a good idea in theory, but you incur a large technical cost to managing the scaling of this service. Additionally if you host your own parser, its accuracy will eventually lag behind public parsing APIs as they include new research results in the coming years. 

For these reasons, I recommend a paid parsing service—I’ve specifically had great luck with Google’s NL API. A parsing service is more expensive than raw cloud compute time, but probably not if you include the lost development time spent managing and debugging your own microservice.


><i class="icon-info"></i>**Extra:** If you’re feeling bold, here’s containerized code to run a TensorFlow-based simple parsing service that you can customize.
> <i class="icon-download"></i> [tf-parser](github.com)

><i class="icon-info"></i>**Extra:** TODO constituency parse link

###Part of Speech Weighting
It's often useful to 


### Dependency Parse Tricks

bigram and trigram trick

### Word Tokenization

### Word2Vec

### Wordnet Annotations

It’s often useful to group words together so that a group can be dealt with by manually or automatically encoding a single rule (remember: most compact truthful separation of data is the goal). For example you might want to automatically generate a rule for a bot to treat “speak + ANY LANGUAGE” as the same concept. 

Wordnet can tell us the ontology of a word (ie. pug -> dog -> animal -> living-entity or shampoo -> bath product -> commerce item -> physical entity). With the previously mentioned feature framework, it is easy to add the group annotations to the annotation list so we can make complex features around groups.

<wordnet.py>

###Sentence is Request?



###Splitting Conversations into Related Groups

###Tracking Bot State

Ideally the next response a bot gives should depend on both the syntax of what the user said last and the current information the bot has. Here's my favorite framework for determining what the bot should say next given these two pieces of information.

Assume we have a "state graph" like the one below. Don't worry how we got it. We'll handle how to train it later. At each node, the bot says something. Each statement the user makes will move us along an edge. Each edge has a syntax group associated with it.
...

Assume we are at a node X. The user says something. First assign a probability that the user's syntax belongs to each of the syntax classes in our language model (discussed above TODO). Turn that vector of language classes into 


##Flows

###Bot
include all my python code?

###Targetting Relevant Content to Users
Assume you have a large amount of content, ads, products, etc that you want to target to the right user at the right time. Here's my favorite framework for handling that. 

Generate User Features:
  - guessing ethnicity, gender, age group from name
  - location from IP address
  - user syntax features
Generate Content Features:
  - creator
  - syntax features

unigram and bigram features (might want to rename)

Collaborative Filtering Data:

After a certain number of clicks (generally less than 100), user and content features aren't useful anymore. The collaborative filtering data tells you all of the information covered in the user and content features. The user and content features and however very useful for new content or users.

Adding Behavior:
Assume for example that you want to make sure a person sees a variety of content. This framework is nice for development scalability because it allows you to add behavior like this.

Common Mistake:
If you don't use this approach, make sure that you account for positive feedback loops. 


###Twitter ISIS Recruitment Spotting
Assume that you want to spot ISIS recruiters on Twitter. The difficulty here is you might not have a dataset of examples to train a model from. So what do you do? 

Don't overthink it. Start with an intentionally broad net. For example search for any tweet with the following words: isis, daesh, or death + america.

The key is search speed so we don't want to do anything even remotely slow (ie. parsing dependencies, word2vec).

This list will produce far more false positives than true positives. However, we can expect nearly all recruitment accounts to have at least one tweet in this list. We need at least 0.5% of Tweets in our list to be true positives for us to feasibly refine our search criteria (a person can reasonably look through 1000 lines in an hour and we want at least 5 positives for the next step).

If the list didn't contain any true positives, we could either choose different search terms or refine the search further.

Once you have these 5 true positives, you can either look through another 10k lines or refine your search. I think it's generally a faster long-term strategy to make your search as refined as possible without utilizing complex, slow search techniques.

Here's an ipython-based flow I made for this example.

NOTE: I SHOULD DO THE CODE BEFORE I WRITE ABOUT OPTIMAL WAY TO DO THIS

> Extra:
> This same technique can be used for other tasks like finding good candidates on Linkedin.



