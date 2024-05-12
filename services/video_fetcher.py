import requests

def extract_video_data(videos):
    extracted_data = []
    for video in videos:
        comment_authors = list(set(
            comment['author']['name'] for comment in video.get('comments', [])
            if 'author' in comment and 'name' in comment['author']
        ))

        video_info = {
            "id": video.get('id', 'Unknown ID'),
            "name": video.get('name', 'No Name Provided'),
            "description": video.get('description', 'No Description Provided'),
            "comment_authors": comment_authors
        }
        extracted_data.append(video_info)
    return extracted_data


def fetch_videos():
    base_url = "http://localhost:8080/videominer/videos"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        video_data = extract_video_data(response.json())
        
        return video_data
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")  
    except Exception as err:
        print(f"An error occurred: {err}")  

    return None
