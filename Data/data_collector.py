from gnews import GNews


def collect_data(
                 language="en",
                 country="US",
                 BATCH_SIZE=100,
                 period=None,
                 start_date=None,
                 end_date=None,
                 exclude_websites=None,
                 proxy=None,
                 max_no_articles=1500
                 ):
    """ Function collects a selection of articles which will be used to train the model

    Args:
    language: The language in which to return results, defaults to en (optional)
    country: The country code of the country you want to get headlines for, defaults to US
    max_results: The maximum number of results to return. The default is 100, defaults to 100
    period: The period of time from which you want the news
    start_date: Date after which results must have been published
    end_date: Date before which results must have been published
    exclude_websites: A list of strings that indicate websites to exclude from results
    proxy: The proxy parameter is a dictionary with a single key-value pair. The key is the
    protocol name and the value is the proxy address

    Output:
    articles: a dictionary of articles containing the articles title, description, published date, url and publisher
    """

    # Initialize some variables
    news = []
    total_articles = 0
    # Instantiate GNews
    gnews_us = GNews(language=language,
                     country=country,
                     max_results=BATCH_SIZE,
                     period=period,
                     start_date=start_date,
                     end_date=end_date,
                     exclude_websites=exclude_websites,
                     proxy=proxy)

    while total_articles < max_no_articles:  # Fetch until you reach your desired total
        articles = gnews_us.get_news_by_topic("BUSINESS")
        news.extend(articles)  # Extend the news list with articles from the current batch
        total_articles += len(articles)

        if len(articles) <= 1:
            # If the fetched batch is smaller than 1, it means there are no more articles to fetch.
            break

    return news


def get_articles(
                 language="en",
                 country="US",
                 period="0.25h",
                 max_results=50,
                 exclude_websites=None,
                 proxy=None,
                 ):
    """ Function gets news from within last 15min, ready to be fed to model.

    Args:
    language: The language in which to return results, defaults to en (optional)
    country: The country code of the country you want to get headlines for, defaults to US (optional)
    max_results: The maximum number of results to return. The default is 10, defaults to 10
    period: The period of time from which you want the news
    exclude_websites: A list of strings that indicate websites to exclude from results (optional)
    proxy: The proxy parameter is a dictionary with a single key-value pair. The key is the
    protocol name and the value is the proxy address (optional)

    Output:
    articles: A dictionaries of article containing the article's title, description, published date, url, and publisher
    title: A list of the new articles titles
    description: A list of the new articles descriptions
    """

    articles = []
    total_articles = 0
    # Instantiate Gnews
    new_articles = GNews(language=language,
                         country=country,
                         period=period,
                         max_results=max_results,
                         exclude_websites=exclude_websites,
                         proxy=proxy)

    while total_articles < max_results:
        print(f"Fetching batch {total_articles + 1}...")
        batch = new_articles.get_news_by_topic("BUSINESS")
        print(f"Fetched {len(batch)} articles in this batch.")
        articles.extend(batch)
        total_articles += len(batch)

    title = []
    description = []
    url = []

    for article in articles:  # Iterate over the list of articles
        title.append(article["title"])
        description.append(article["description"])
        url.append(article["url"])

    return articles, title, description, url
