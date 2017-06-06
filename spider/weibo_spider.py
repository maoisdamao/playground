import re
import requests


def request_data(comment_api, page=1):
    r = requests.get(api.format(page=page))
    r = r.json()
    return r


if __name__ == "__main__":
    # replace {post_id} to your post id
    api = "https://m.weibo.cn/api/comments/show?id=post_id&page={page}"
    current_page = 1
    res = request_data(api, page=current_page)
    comments = res['data']
    print("page:", current_page)
    total = int(res['total_number'])
    rest_counts = total
    while rest_counts > 0:
        current_page += 1
        comment_api = api.format(page=current_page)
        r = request_data(comment_api, page=current_page)
        # for weibo total bug
        if not r['ok']:
            break
        print("page:", current_page)
        comments.extend(r['data'])
        rest_counts -= len(r['data'])

    # following part deals with json data format
    # reply_text exists means it is a comment on another comment
    comments = [i for i in comments if "reply_text" not in i.keys()]
    # sort comments on number of likes
    comments = sorted(comments, key=lambda i: i['like_counts'], reverse=True)
    print("total valid comments:", len(comments))
    comments_text = []
    with open("./comments.txt", 'wt') as f:
        for item in comments:
            # remove weibo emoji
            text = re.sub(r'<.*>', '', item['text'])
            print(text+'\n', file=f)
