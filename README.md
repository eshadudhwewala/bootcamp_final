# Predicting Wine Ratings
This is my final project on predicting wine ratings. 

Wine is a widely enjoyed alcoholic beverage produced in many countries around the world. For my final project, I plan to build a predictive model that can accurately estimate the Wine Rating of various red wines, which differ across factors like price, country or region of origin, and vintage year. I will explore the trends and patterns in these ratings to offer meaningful insights into what contributes to a higher wine rating. These ratings, ranging from 0 to 5, are assigned by users on Vivino.com and serve as a useful indicator of a wine’s taste and overall quality.

Such a model can be particularly valuable for individuals looking to purchase wine but unsure where to begin, or those seeking the best value for a given price point. The predictions will help guide more informed financial decisions, ensuring buyers select wines that are likely to be enjoyable and worth the cost. In addition, wine retailers and marketers can leverage these insights to better position their products, highlight high-performing wines, and tailor marketing strategies to align with consumer preferences. Ultimately, this model aims to give wine buyers actionable insights and greater confidence in their choices.

The data for this project comes from Kaggle in CSV format and offers detailed information on a range of wines from Vivino.com. The dataset will need some cleaning and wrangling due to potentially irrelevant columns and the presence of null values. Building an accurate regression model may pose some challenges because of the data's inherent volatility—wine ratings are purely based on user opinions, which are subjective by nature. Still, I expect that certain variables will have predictive power and can help estimate wine ratings with reasonable accuracy.

This dataset provides detailed information on various wines and their ratings, including attributes like name, country of origin, region, winery, price, and vintage year. It consists of 8,666 rows and 8 columns, which I can use as features in predicting wine ratings.

The dataset I will be working with contains detailed information on 8,666 red wines from around the world, with vintages ranging across multiple years from 1988 to 2019. The wines vary in price from as low as $3.55 to over $3,400, and have received between 25 and more than 20,000 ratings between 2.5 and 4.8 on Vivino.


Based on the distribution from my graph in Python, wine ratings appear to follow a roughly normal distribution centered around 3.9 to 4.0. While most wines cluster within this range, ratings span from a minimum of approximately 2.5 to a maximum close to 4.8 out of 5.

My visualizations also show that the majority of wines in the dataset were made after 2005, with a notable concentration in the 2010s. While high ratings appear across all years, older vintages tend to be more sparsely represented but still show consistently strong ratings. This may suggest that more recent wines dominate the market in quantity, while older wines, though fewer, still hold high perceived quality. 

Most wines in the dataset have relatively few ratings, with a sharp drop-off beyond 2,000 reviews. While highly-rated wines appear across all ranges of review counts, there is no strong pattern suggesting that wines with more ratings are rated significantly higher or lower. This may indicate that popularity (as measured by number of ratings) does not necessarily correlate with perceived quality.

While most countries have average ratings clustered around 3.8 to 4.1, a few stand out with notably higher or lower scores. Countries like Switzerland, Slovenia, and Georgia show especially high average ratings, whereas countries such as China, Bulgaria, and Canada have lower average ratings in comparison. This suggests that while quality is fairly consistent across many regions, there are some clear differences in perceived wine quality based on country of origin.


While most wines are priced below  500, a few outliers reach over 3,000. Interestingly, high ratings are observed across nearly the entire price range, suggesting that spending more does not always guarantee a better-rated wine. In fact, many lower-priced wines receive ratings comparable to those of high-end bottles, indicating that good quality can be found at a variety of price points. However, none of the wines priced at 500 or higher have ratings below 5 stars, showing that the more expensive wines do tend to have higher rather than lower ratings.

To predict wine ratings, I implemented several different regression models to determine which one performs best in capturing score variability and explaining the patterns in the data. For each model, I applied an 80-20 train-test split—training the model on 80% of the dataset and evaluating its performance on the remaining 20%.

I assessed the performance of each model by comparing key metrics, such as mean squared error, against a baseline model. This baseline was established by predicting the average wine rating across the entire dataset.

I then chose to build a multiple regression model to use several independent variables to predict the dependent variable, wine rating, as I believed these factors might collectively impact the score. Multiple linear regression enabled me to analyze both the individual relationships and the combined influence of these predictors on the rating. Overall, my multiple regression model outperformed the baseline model. Both the training and testing mean squared errors (0.070 and 0.066, respectively) were lower than the baseline MSE of approximately 0.095, with the training set showing slightly better performance. This suggests that the model was able to capture meaningful patterns in the data and account for variation in wine ratings using the available predictors, rather than simply predicting the average score.

In terms of feature importance, the most influential variable in predicting wine ratings was Price, followed by Year and Country. This indicates that higher wine prices tend to carry more predictive power in estimating ratings, while the year of production and country of origin contribute to a lesser, but still noticeable, extent.
I decided to experiment with k-nearest neighbors (KNN) regression next, as this method predicts outcomes based on the similarity between data points in the feature space. Since some of my initial visualizations suggested the presence of local patterns or clusters of similar wines with shared characteristics, KNN seemed well-suited to capture these localized relationships in the data.

My KNN regression model outperformed both the baseline and the multiple linear regression model. While the model performed slightly better on the training set (MSE ≈ 0.035) than on the testing set (MSE ≈ 0.038), the test performance still represented a notable improvement over previous approaches. I believe this is because KNN is capable of capturing non-linear patterns and local relationships in the data—something that may be especially relevant for wine ratings.

The strong performance may also be attributed to hyperparameter tuning through grid search, which allowed me to identify the optimal number of neighbors for the model, ultimately improving its accuracy.

In this model, Price emerged as by far the most important predictor, followed by Country and Year. This suggests that wine price continues to be the dominant factor in predicting ratings, while regional and vintage information contribute to a lesser, yet meaningful, extent.

I also chose to build a decision tree regression model because, like k-nearest neighbors, it can capture non-linear relationships within the wine rating data. In addition, decision trees offer a clear and interpretable structure, making it easy to understand how predictions are made based on different feature values. This provides valuable insight into the factors that most strongly influence wine ratings.

While the decision tree regression model performed better than both the baseline and the multiple linear regression model, its performance was slightly weaker than the KNN model. I suspect this is due to the low maximum depth selected for the tree (depth = 3), which may have limited its ability to capture more complex patterns in the data. With such a shallow tree, the model relied almost entirely on Price for its decisions, while completely ignoring Year and Country, which may have affected its predictive accuracy.

In this model, Price was once again the most influential predictor of wine ratings—by a wide margin. Unlike in previous models, however, Year and Country were assigned zero importance, indicating that the tree did not consider them relevant in its final structure. This could be a result of the tree's depth constraint, which prevented it from exploring more nuanced interactions among features.

For my final model, I chose to expand on the decision tree approach by building a random forest regression model. Given that the single decision tree showed promising results, I wanted to explore ensemble methods like random forests, which aggregate the predictions of multiple trees to enhance overall predictive accuracy.

Overall, my random forest model delivered the strongest performance out of all the models I tested. Although there was a slightly larger gap between the training and testing mean squared errors compared to some previous models, the random forest achieved the lowest MSE on the test set (≈ 0.036), indicating it was the most effective at predicting wine ratings.

In terms of feature importance, Price remained the most significant predictor by a large margin, followed by Country (specifically Spain and Australia). Interestingly, many other countries—including traditionally prominent wine producers—were assigned zero or even negative importance, as was Year, which suggests that the model found little value in these variables for improving prediction accuracy.

In my analysis of wine ratings, all the models I constructed demonstrated improved performance over the baseline predictor, highlighting their predictive value. The models ranked by performance are as follows: Random Forest Regression, K-Nearest Neighbors Regression, Decision Tree Regression, and Multiple Linear Regression.

Key Findings:

Success of the Random Forest Model: The Random Forest model achieved the best overall performance, with the lowest mean squared error on the test data among all models. Its strong results suggest that ensemble methods are particularly well-suited for capturing complex, non-linear relationships within the wine rating data.

Most Impactful Feature – Price: Across all models, Price consistently emerged as the most influential variable in predicting wine ratings. It played a dominant role in both the KNN and tree-based models, suggesting a strong association between wine cost and perceived quality.

Variable Influence – Country and Year: While Country showed some predictive power in select models (notably in the Random Forest), its importance varied and was often low or even zero. Year, surprisingly, had little to no predictive value across all models. This indicates that consumers may not weigh vintage year as heavily when rating wines, at least within this dataset.

Conclusion: The Random Forest model, through its ensemble of decision trees, proved especially effective at capturing nuanced patterns in the data. The consistent dominance of Price as a predictor suggests that it serves as a strong proxy for perceived wine quality. Meanwhile, other factors like Country and Year offered limited predictive insight, and their importance fluctuated depending on the model. These findings provide useful direction for future predictive modeling in consumer rating data, offering a clearer picture of which wine characteristics most influence user ratings.




Next Steps/Improvements
To further enhance the predictive power of my models and gain deeper insights into what drives wine ratings, I would consider incorporating the following additional features or strategies into future versions of this project:

Grape Variety and Wine Type ○ Including data on grape varieties (Pinot Noir, Cabernet Sauvignon) and specific wine types (blend, single varietal) could offer a clearer picture of how certain flavors and styles are perceived by reviewers.

Tasting Notes or Descriptions ○ Applying text analysis or natural language processing (NLP) to wine tasting notes or user reviews from platforms like Vivino could uncover keywords or descriptors ("oak," "bold," "fruity") that influence higher ratings.

Winery Reputation and Awards ○ Incorporating winery-level features—such as awards, brand reputation, or critic scores—could help account for external credibility factors that may impact ratings beyond price or region.

Consumer Demographics ○ If available, incorporating demographic information about the reviewers (age, location, experience level) could reveal how personal preferences and regional biases affect scoring patterns.

By integrating these additional data points, I could refine the models to capture a more comprehensive view of what drives wine ratings. This would not only enhance model accuracy but also offer valuable insights to winemakers, marketers, and wine retailers looking to better understand consumer preferences and improve product positioning.







