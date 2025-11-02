import re

def remove_hashtags(text: str) -> str:
    """Removes entire hashtags (e.g., '#sarcasm') from a string."""
    return re.sub(r'#\w+', '', text).strip()

def integrate_hashtags(text: str) -> str:
    """
    Integrates hashtags into the text by removing the '#' and splitting
    CamelCase words.
    """
    def split_camel_case(match):
        hashtag = match.group(1)
        return ' ' + ' '.join(re.findall(r'[A-Z][a-z]*|[a-z]+', hashtag))

    return re.sub(r'#(\w+)', split_camel_case, text)

def main():
    """Example usage of the preprocessing functions."""
    # Raw tweet
    tweet = "@USER Canada doesn’t need another CUCK! We already have enough #LooneyLeft #Liberals f**king up our great country! #Qproofs #TrudeauMustGo	OFF"

    tweet_without_hashtags = remove_hashtags(tweet)
    print(f"Tweet without hashtags: '{tweet_without_hashtags}'")

    tweet_with_integrated_hashtags = integrate_hashtags(tweet)
    print(f"Tweet with integrated hashtags: '{tweet_with_integrated_hashtags}'")

if __name__ == "__main__": 
    main()