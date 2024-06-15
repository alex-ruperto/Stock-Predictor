from pytrends.request import TrendReq
import pandas as pd

# Fetch Google trends data for a specific keyword and timeframe
def fetch_google_trends(keyword, timeframe='2020-01-01 2023-12-31'):
    # Initialize a TrendReq object with English as the language and a timezone offset of 360 minutes (UTC+6)
    pytrends = TrendReq(hl='en-US', tz=360)

    # Build the payload with the given keyword, category (0 for all categories), the specified timeframe, 
    # geographic area (empty string for worldwide), and Google property (empty for web search)
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')

    # Fetch the interest over time for the specified keyword. This returns a DataFrame
    trends_data = pytrends.interest_over_time()

    # Check if the returned DataFrame is not empty
    if not trends_data.empty:
        # Drop the 'isPartial' column which indicates whether the data is potentially incomplete for the latest periods
        trends_data = trends_data.drop(labels=['isPartial'], axis='columns')

        # Rename the column with the keyword trend data to include the keyword in its name for clarity
        trends_data.rename(columns={keyword: f'{keyword}_trend'}, inplace=True)

    # Return the processed DataFrame
    return trends_data
    

# Example usage of the function to fetch Google Trends data
keyword = 'AMC Stock'
timeframe = '2020-01-01 2020-12-31' # Define the timeframe for the Trends data
google_trends_data = fetch_google_trends(keyword, timeframe) # Call the function with the keyword and timeframe

print(google_trends_data.head(5))

''' How to interpret these values:
The value is scaled from 0 to 100, where 100 represents the peak popularity for the term.
A value of 12 means that during the week of 2020-01-05, the search interest for the keyword was at 
12% of its peak popularity.
Similarly, a value of 10 for the week of 2020-01-12 indicates that the search interest dropped to 
10% of the peak popularity during that week.
The consistent values of 9 for the weeks following suggest a stable but lower search interest 
relative to the keyword's peak popularity during the specified timeframe.
'''
