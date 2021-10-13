#Constants
IMG_SIZE = 48
EMOTIONS = ["afraid", "angry", "disgust", "happy", "neutral", "sad", "surprised"]

HF_PATH = 'haarcascade_frontalface_default.xml'
BASE_URL = "https://api.spotify.com/v1/recommendations"
MARKET = "IN" #market from which songs are recommended
SA = "4NHQUGzhtTLFvgF5SZesLK" #seed artist for spotify (this is not a constant will update as per the genre)
SA_D = {"hiphop": "7dGJo4pcD2V6oG8kP0tJRR", # eminem
        "rock": "31hrPUMBg96szrqNAb3oqP", # blacklite district
        "country": "1UTPBmNbXNTittyMJrNkvw", # Blake Shelton
        "pop": "5IH6FPUwQTxPSXurCrcIov", # alec benjamin
        "metal": "2ye2Wgw4gimLv2eAKyk1NB", # metallica
        "disco": "4tZwfgrHOc3mvqYlEYSvVi", # daft punk
        "reggae": "6BH2lormtpjy3X9DyrGHVj", # bob marley
        "jazz": "1Mxqyy3pSjf8kZZL4QVxS0", # frank sinatra
        "bollywoodpop": "6CXEwIaXYfVJ84biCxqc9k", # vishal dadlani
        "classical": "3WrFJ7ztbogyGnTHbHJFl2", # beatles
        "blues": "3WrFJ7ztbogyGnTHbHJFl2", #B. B king
        }
ST = "0c6xIDDpzE81m2q797ordA" #seed track for spotify
ST_D = {
    "hiphop": "7MJQ9Nfxzh8LPZ9e9u68Fq",
    "jazz": "0elmUoU7eMPwZX1Mw1MnQo",
    "rock": "1DWiVxo482tHbgTWKHMWqg",
    "metal": "1hKdDCpiI9mqz1jVHRKG0E",
    "country": "0cB74Rrq9gKE5iUjwG9raA",
    "reggae": "4dbaWokGcqEWvwTZDBbMD3",
    "disco": "2cGxRwrMyEAp8dEbuZaVv6",
    "pop": "1xQ6trAsedVPCdbtDAmk0c",
    "classical": "7pKfPomDEeI4TPT6EOYjn9",
    "blues": "3cg0dJfrQB66Qf2YthPb6G"
}
TOKEN = "BQBUBVI6ndX-R2t70WMCz3H1kjFNY7Pih0e9wg4kyjW-k-V3kHAlTIxxneT3CiM9-zXakDi9-qFk3YLkMspWbtWZ1NttlcZo6pvebJJZvMQMZkXovgHgenlqWADoLdkEZer_MFsgu3wV9fqFjvi4rNd7-x0IWYyYWyI"
HEADER = {
    "content-Type": "application/json"
}
HEADER["authorization"] = "Bearer "+TOKEN
