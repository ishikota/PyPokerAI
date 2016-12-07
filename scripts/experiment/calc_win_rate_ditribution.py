import pickle
import time
import multiprocessing
from multiprocessing import Process, Queue

import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt

from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.engine.deck import Deck
from pypokerengine.engine.card import Card

CPU_NUM = multiprocessing.cpu_count()

def find_good_number_of_simulation():
    for i in range(5, 16):
        simulation_num = i * 100

def generate_holecard_ranking(nb_entry):
    nb_simulation = 1000
    deck = Deck()
    ranking = []
    for _ in range(nb_entry):
        deck.shuffle()
        hole = deck.draw_cards(2)
        win_rate = estimate_hole_card_win_rate(nb_simulation, 10, hole)
        ranking.append((win_rate, hole))
        deck.restore()
    return sorted(ranking)[::-1]

def generate_holecard_ranking_mp(nb_entry):
    queue = Queue()
    nb_process = 4
    st = time.time()
    def task(q):
        ranking = generate_holecard_ranking(nb_entry)
        q.put(ranking)
    ps = [Process(target=task, args=(queue,)) for _ in range(nb_process)]
    [p.start() for p in ps]
    [p.join() for p in ps]
    print "Finished nb_process=%d, nb_entry=%d with time=%s" % (
            nb_process, nb_entry, time.time() - st)
    ranking = []
    # queue.qsize() raises NotImplementedError on MacOS
    for i in range(nb_process):
        try:
            each_ranking = queue.get()
        except:
            break
        ranking += each_ranking
        print "fetched item %d from queue" % (i+1)
    return sorted(ranking)

def subtask_hole_community_winrate(q, community_num, nb_sample):
    print "task started"
    nb_simulation = 1000
    deck = Deck()
    res = []
    for i in range(nb_sample):
        deck.shuffle()
        hole = deck.draw_cards(2)
        community = deck.draw_cards(community_num)
        win_rate = estimate_hole_card_win_rate(nb_simulation, 10, hole, community)
        res.append((win_rate, hole, community))
        deck.restore()
    q.put(res)
    print "task finished"

def search_distribution_of_win_rate_with_community(community_num):
    queue = Queue()
    nb_process = multiprocessing.cpu_count()
    st = time.time()
    ps = [Process(target=subtask_hole_community_winrate, args=(queue, community_num, 2500))
            for _ in range(nb_process)]
    [p.start() for p in ps]
    return queue

def save_distribution_of_win_rate_with_community(community_num, queue):
    res = queue.get()
    res += queue.get()
    res += queue.get()
    res += queue.get()
    print "community_num=%s, length of res==%s" % (community_num, len(res))
    with open("community_num_%d.pickle" % community_num, "wb") as f:
        pickle.dump(res, f)
    win_rates = [r for r, h, c in res]
    plt.hist(win_rates, bins=100)
    plt.title("distribution of win rate (community=%d)" % community_num)
    plt.xlabel("win rate")
    plt.ylabel("frequency")
    plt.savefig("community_num_%d.png" % community_num)
    plt.clf()


def subtask_holecard_winrate(q, ids, start, end):
    nb_simulation = 1000
    my_task = ids[start:end]
    print len(my_task)
    res = []
    for id1, id2 in my_task:
        hole = [Card.from_id(i) for i in [id1, id2]]
        win_rate = estimate_hole_card_win_rate(nb_simulation, 10, hole)
        res.append((win_rate, hole))
    q.put(res)

def calc_all_holecard_winrate():
    queue = Queue()
    idxs = [[(i,j) for j in range(i+1, 53)] for i in range(1, 53)]
    idxs = [item for sublist in idxs for item in sublist]
    base = 1326/4
    p1 = Process(target=subtask_holecard_winrate, args=(queue, idxs, base*0, base*1))
    p2 = Process(target=subtask_holecard_winrate, args=(queue, idxs, base*1, base*2))
    p3 = Process(target=subtask_holecard_winrate, args=(queue, idxs, base*2, base*3))
    p4 = Process(target=subtask_holecard_winrate, args=(queue, idxs, base*3, base*4+2))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    res = []
    res += queue.get()
    res += queue.get()
    res += queue.get()
    res += queue.get()
    with open("all_holecard_winrates.pickle", "wb") as f:
        pickle.dump(res, f)
    win_rates = [w for w,c in res]
    plt.hist(win_rates, bins=100)
    plt.title("distribution of win rate of all holecard")
    plt.xlabel("win rate")
    plt.ylabel("frequency")
    plt.savefig("all_holecard_winrates.png")


def research_distribution_of_holecard_win_rate():
    nb_entry = 5000
    nb_entry4task = (int)(nb_entry/CPU_NUM)
    ranking = generate_holecard_ranking_mp(nb_entry4task)
    win_rates = [r for r, c in ranking]
    plt.hist(win_rates)
    plt.savefig("research_distribution_of_holecard_win_rate.png")
    with open("research_distribution_of_holecard_win_rate.txt", "wb") as f:
        f.write(str(ranking))

