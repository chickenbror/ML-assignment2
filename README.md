Assignment 2 from the course LT2222 in the University of Gothenburg's spring 2021 semester.

*Name: Hsien-hao Liao*   


**Part 1 preprocessing**/
In the preprocessing of the tokens, I originally skipped the punctuation, but it somehow removed some NEs as a result (fewer than 6922), so I only removed the non-NE tokens that are punctuation. I also decided to only skip commas, full stops, parentheses, etc, as I considered some tokens like '%' to be useful context. I originally used the default config of WordNetLemmatizer, which assumed everything was a noun, but words like 'has' became 'ha' as a result. So I used the POS tags to make it output the correct lemma forms.

**Part 2 confusion matrix analysis**/
I took the recommendation from the Discord discussions and added 5 start/end tokens before and after each sentence so that the features length is always 10. It also makes the before/after context distinguishable, ie, the first 5 are 'before' and the last 5 are 'after'.

**Part 3 vectors table** /
I used the tf-idf vectorizer to vectorize each 'features' list to account for each feature's weight, and then used TruncatedSVD to reduced the number of dimensions (choice of dims size noted below in Part 5).

**Part 5 confusion matrix analysis**/
The chance of successful classification appears to be relative to the proportional size of each NE class in the data. For example, 'geo', 'org', 'gpe' and 'per' were by a big margin correctly classified. Also, the training data performed better than the test data, where NE classes that account for small proportions in the data got few to none successful classification. One thig worth noting is that the dims size in Part 3 seems to matter. Originally I reduced them to 300-D, then 1000-D, and then 3000-D. The latter took longer time to process but it gave eg 'art', 'eve' more correct classifications, in particular in the training ddata confusion matrix.
