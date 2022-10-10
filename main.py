from utils import getComments, getGregPosts, getSubPosts
import time
from collections import defaultdict
import json
import pprint
import heapq
import random
import copy
ind = 0
start = 0
split_size=1
gpu_ind = 2
subs = ['CasualConversation', 'Advice', 'truegaming', 'writing', 'fitness', 'TalesFromRetail', 'LetsTalkMusic', 'books', 'movies', 'college']
#sub=subs[9]
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import spacy
import postfilter

model_bert = BertForNextSentencePrediction.from_pretrained("bert-large-uncased").to(f"cuda:{gpu_ind}")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
nlp = spacy.load("en_core_web_sm") #python -m spacy download en_core_web_sm
original = 0
response = 0
random.seed(42)


def getScore(prompt, i):
    encoding = tokenizer(prompt, i, return_tensors="pt", truncation=True).to(f"cuda:{gpu_ind}")
    outputs = model_bert(**encoding, labels=torch.LongTensor([1]).to(f"cuda:{gpu_ind}"))
    logits = outputs.logits
    return_val = float(logits[0,0] - logits[0,1])
    del encoding
    del outputs
    return return_val

def getMostLikelyBeamSearch(curr_state, options, n, response_d):
    global original, response
    prompt = " ".join(curr_state)
    results = []
    for i in options:
        diff = getScore(prompt, i)
        results.append((diff, i))
    best = heapq.nlargest(n, results)
    best2=[]
    for i in best:
        if i[1] == options[-1]:
            best2.append(i + ("", "_original"))
            continue
        results2 = []
        results2.append(i + ("", "_original",))
        for x in response_d[i[1]]:
            if x == "" or x == "[deleted]":
                continue
            if x[-1] not in [".", "!", "?"]:
                x = x + "."
            if x in prompt:
                continue
            diff = getScore(prompt, i[1] + '. ' + x)*(1.15 - 0.01 * len(x.split(" ")))
            results2.append((diff, i[1], x, "_response"))
        best_curr = heapq.nlargest(1, results2)[0]
        while best_curr[3] == "_response":
            curr_opt = best_curr[1][:-1]
            opt_sentences = nlp(curr_opt)
            split_opts = [str(x).strip() for x in opt_sentences.sents]
            for j in curr_state:
                breakTime = False
                for k in split_opts:
                    if k in j:
                        results2.remove(best_curr)
                        print("Removing redundant threading")
                        try:
                            best_curr = heapq.nlargest(1, results2)[0]
                        except:
                            try:
                                best_curr = results2[0]
                            except:
                                best_curr = i + ("", "_original1",)
                        breakTime = True
                        break
                if breakTime:
                    break
            break
        if best_curr[3] == "_original":
            original += 1
        elif best_curr[3] == "_response":
            response += 1
        best2.append(best_curr)
    return best2

def main_program_logic(prev_state, n, response_d):
    global nlp
    curr_state = prev_state[0]
    replies = prev_state[1]
    split_post = prev_state[2]
    comments = prev_state[3]
    i = prev_state[4]
    agg_score = prev_state[6]
    hasReaction = prev_state[7]
    if i >= len(split_post) - 2:
        return -1

    curr_sent = split_post[i]
    if curr_sent.strip() == "":
        i += 1
        try:
            curr_sent = split_post[i+1]
        except:
            return -1
    if hasReaction:
        curr_state[-1] += ' ' + curr_sent
        hasReaction = False
    else:
        curr_state.append(curr_sent)
    next_n = getMostLikelyBeamSearch(curr_state, comments + [split_post[i+1]], n, response_d)
    new_prev_states = []
    for next in next_n:
        n_curr_state = copy.copy(curr_state)
        n_replies = copy.copy(replies)
        n_comments = copy.copy(comments)
        n_i = i
        n_next = next
        n_hasReaction = False
        n_score = agg_score
        while n_i + 1 < len(split_post) and n_next[1] == split_post[n_i + 1]:
            n_curr_state[-1] += ". " + n_next[1]
            n_i += 1
            n_score += n_next[0]
            try:
                n_next = getMostLikelyBeamSearch(n_curr_state, n_comments + [split_post[n_i+1]], 1, response_d)[0]
            except:
                new_prev_states.append([n_curr_state, n_replies, split_post, n_comments, n_i+1, next, n_score, n_hasReaction])
                break
        try:
            text_sentences = nlp(n_next[1])
            split_next = [str(x).strip() for x in text_sentences.sents]
            if type(split_next) == list:
                for opt in split_next:
                    if opt == "":
                        continue
                    to_remove = []
                    for remaining in n_comments:
                        if opt in remaining:
                            to_remove.append(remaining)
                    for opt_remove in to_remove:
                        try:
                            n_comments.remove(opt_remove)
                        except:
                            continue
            n_comments.remove(n_next[1])
        except:
            pass
        n_replies.append(n_next[1])
        n_curr_state.append(n_next[1])
        try:
            if n_next[3] != "_original":
                n_hasReaction = True
                n_curr_state.append(n_next[2])
                n_comments.remove(n_next[2])
        except:
            pass
        new_prev_states.append([n_curr_state, n_replies, split_post, n_comments, n_i+1, n_next, n_score + n_next[0], n_hasReaction])
    return new_prev_states

def beam_end_condition(beams_completed):
    for bc in beams_completed:
        if bc != -1:
            return False
    return True


def threading(num_posts, num_threads):
    start = time.time()
    # Number of posts to process
    k = num_posts
    tot_turns_generated = 0
    posts_modified = defaultdict(list)
    prev_state = []
    # Width of beam search
    n = num_threads
    a = getSubPosts(sub)
    running_total_score = 0
    num_gen = 0
    num_turns_generated = 0
    length = float(len(a))/split_size
    
    for post_data in a:
        try:
            num_gen += 1
            print(num_gen)
            all_convo_options = defaultdict(int)
            post = post_data[1]["text"]
            post_sentences = nlp(post)
            split_post = [str(x) for x in post_sentences.sents]
            if len(split_post) == 0:
                print("Split Post was empty, skipping this post")
                continue
            # Force title and first sentence of post to be together
            if post_data[1]["title:"][-1] not in ["!",".","?"]:
                split_post[0] = post_data[1]["title:"] + ". " + split_post[0]
            else:
                split_post[0] = post_data[1]["title:"] + " " + split_post[0]

            comments, response_d = getComments(post_data[1])
            print(len(comments))
            if comments == []:
                print("No comments")
                continue
            if len(comments)>1000:
                    print("Cutting down to 1000 comments")
                    comments = random.sample(comments, 1000)
            if len(comments) < 1.5 * len(split_post):
                continue
            curr_state = []
            i = 0
            j = 0
            replies = []
            prev_state = [[curr_state, replies, split_post, comments, i, 0, 0.0, False]]
            prev_states = main_program_logic(prev_state[0], n, response_d)
            if prev_states == -1:
                print("Prev states was -1 on initial post")
                continue
            all_convo_options[j] = prev_states
            j += 1

            while not beam_end_condition(prev_states) and j < 10:
                total_beam_options = []
                for i in range(n):
                    if prev_states[i] == -1:
                        continue
                    nexts = main_program_logic(prev_states[i], n, response_d)
                    if nexts == -1:
                        total_beam_options.append((-1, "bad"))
                        continue
                    for option in nexts:
                        total_beam_options.append((option[-1], option))
                highest_scores = heapq.nlargest(n, total_beam_options)
                prev_states = []
                for chosen in highest_scores:
                    if chosen[0] == -1:
                        continue
                    prev_states.append(chosen[1])
                if len(prev_states) == 0:
                    break
                all_convo_options[j] = prev_states
                j += 1

            size_d = len(all_convo_options)
            num_turns_generated += size_d * 2
            post_data[1]["response"] = all_convo_options[size_d-1][0][1]
            dialogue = []
            curr_state = all_convo_options[size_d-1][0][0]
            score = all_convo_options[size_d-1][0][-2]
            for i, l in enumerate(curr_state):
                curr_state[i] = l.strip().replace("\n", "")
            post_data[1]["dialogue"] = curr_state
            post_data[1]["score"] = score
            del post_data[1]["comments"]
            posts_modified[post_data[0]].append(post_data[1])
            running_total_score += (score) / float(size_d)
            print("#########################################################################################")
            print(num_gen)
            print(f"Original {original} | Response {response}")
            print("#########################################################################################")
            with open(f"./output/{sub}{ind}.json", "w") as data:
                json.dump(posts_modified, data, indent=2, ensure_ascii=False)
        except:
            pass

    end = time.time()
    print(f"Average Score: {running_total_score / float(num_gen)}")
    print(f"Time Elapsed: {end-start}")
    print(f"Time per post processed: {float(end-start)}")
    print(f"Total Turns Generated: {num_turns_generated}")
    print(f"Turns generated per second: { num_turns_generated / float(end-start)}")
    with open(f"./output/{sub}{ind}.json", "w") as data:
        json.dump(posts_modified, data, indent=2, ensure_ascii=False)
    
    filt = []
    #take out lowest scoring 5% of entries
    for i in posts_modified:
        for j in posts_modified[i]:
            filt.append((float(j["score"])/len(j["dialogue"]), j))

    filt.sort(key=lambda x: x[0])
    filt = filt[int(len(filt)*0.05):]
    posts_modified = [x[1] for x in filt]
     

    with open(f"./output/{sub}_nspfiltered{ind}.json", "w") as data:
        json.dump(posts_modified, data, indent=2, ensure_ascii=False)
    
    print(f"Original {original} | Response {response}")
 
if __name__ == "__main__":
    print("Please input the subreddit name (data should be in reddit/subname:")
    global sub
    sub = input()
    print("Please input the number of posts to process:")
    num_posts = int(input())
    print("Please input the number of threads to use:")
    num_threads = int(input())
    threading(num_posts, num_threads)
    postfilter(sub)
    