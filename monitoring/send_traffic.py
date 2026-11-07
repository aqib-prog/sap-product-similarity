"""Send realistic traffic to a running service, so that the dashboard has something to show.

    python monitoring/send_traffic.py --seconds 300

The traffic has four phases, like a normal day in a small shop:
  1. warm-up    a few searches a second, mostly new products (cache misses)
  2. busy       steady traffic, exact and fast mixed, popular products repeat (cache hits)
  3. rush       a burst of many searches a second, mostly fast
  4. cool-down  quiet again
In every phase a few requests have wrong input (422) or an unknown product (404).
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request

SEED = "26d41bdc1495de290bc8e6062d927729"
WEIGHTS = [None, None, None, None, "image:1", "title:1", "text_vectors:0.5,image:0.5"]

# name, share of the time, searches per second, share of fast searches, share of repeats
PHASES = [
    ("warm-up", 0.20, 3.0, 0.1, 0.1),
    ("busy", 0.40, 8.0, 0.5, 0.6),
    ("rush", 0.20, 20.0, 0.8, 0.5),
    ("cool-down", 0.20, 4.0, 0.3, 0.6),
]


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as problem:
        return problem.code, problem.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()

    random.seed(11)
    pool = [SEED]  # every product we have seen in an answer
    sent = {}
    start = time.time()
    for name, share, rate, fast_share, repeat_share in PHASES:
        phase_end = start + args.seconds * sum(
            p[1] for p in PHASES[: PHASES.index((name, share, rate, fast_share, repeat_share)) + 1]
        )
        print(f"{name}: about {rate:g} searches a second")
        while time.time() < phase_end:
            # a repeat asks about a product from the last few answers, so the cache can help
            product = random.choice(pool[-25:] if random.random() < repeat_share else pool)
            roll = random.random()
            if roll < 0.04:  # wrong input
                query = f"product_id={product}&num_similar=0"
            elif roll < 0.055:  # unknown product
                query = f"product_id={'9' * 32}&num_similar=5"
            else:
                query = f"product_id={product}&num_similar={random.choice([5, 10, 10, 20])}"
                if random.random() < fast_share:
                    query += "&mode=fast"
                weights = random.choice(WEIGHTS)
                if weights:
                    query += f"&weights={weights}"
            status, body = get(f"{args.url}/find_similar_products?{query}")
            sent[status] = sent.get(status, 0) + 1
            if status == 200:
                pool.extend(json.loads(body)[:3])
            time.sleep(random.expovariate(rate))
    print("sent, by status code:", dict(sorted(sent.items())))


if __name__ == "__main__":
    main()
