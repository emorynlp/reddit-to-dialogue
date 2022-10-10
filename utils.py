import re
import os
import json
import numpy
from emoji import UNICODE_EMOJI
import pandas
import spacy
import redditcleaner
from collections import defaultdict
import re
from datetime import date
#from choi_code_changes_d import sub
nlp = spacy.load('en_core_web_sm')


def checkpunct(s):
	ss= s.lower()
	for a in s:
		if not a.isalnum() and a not in ".'’,!()?\n -;:~\"&$%+“”*/…—" and a not in UNICODE_EMOJI:
			#print(s,a,'gggggggggggggggg\n\n\n\n\n\nggggggggggggg')
			print(f"bad char:{a}")
			return False
	if " op " in ss or " post " in ss or "edit:" in ss or "tldr" in ss or "post" in ss or "thread" in ss or "reply" in ss or "upvote" in ss or "downvote" in ss:
		print(ss)
		print("Edit or tldr present")
		return False

	return True


def counturls(string):
	# findall() has been used
	# with valid conditions for urls in string
	import time
	if "www." in string or ".com" in string:
		print("url")
		return 1
	a = len([token for token in nlp(string) if token.like_url])
	if a>0: print("url")
	return a


def getSubPosts(sub, dataDir="reddit/"):
	data = dataDir+os.sep()+sub
	post_names = os.listdir(data)
	posts = []
	regex_to_remove = ["\([0-9]*[MmFf]*\)", "[0-9]+[a-zA-Z]+"]
	for i in post_names:
		try:
			a = json.load(open(data + "/" + i))
			posts.append((sub,a))
			posts[-1][1]["text"] = redditcleaner.clean(posts[-1][1]["text"])
			text_sentences = nlp(redditcleaner.clean(posts[-1][1]["text"]))
			c = ""
			for reg in regex_to_remove:
				posts[-1][1]["title:"] = re.sub(reg, '', posts[-1][1]["title:"])
				posts[-1][1]["title:"] = re.sub("  ", ' ', posts[-1][1]["title:"])
			title = posts[-1][1]["title:"].lower()
			count = 0
			for ii in text_sentences.sents:
				count+=1
				iii = str(ii).lower()
				if "tldr" in iii or "edit:" in iii or " op " in iii or " post " in iii or "tl;dr" in iii or "thread" in iii:
					break
				if title == iii:
					continue
				elif count<2 and (title in iii or iii in title):
					continue
				c+=" "
				c+=str(ii)
				for reg in regex_to_remove:
					c = re.sub(reg, '', c)
			posts[-1][1]["text"] = c
			
			if not checkpunct(posts[-1][1]["text"]) or counturls(posts[-1][1]["text"]) > 0 or not checkpunct(posts[-1][1]["title:"]) or counturls(posts[-1][1]["title:"]) > 0 or "thread" in title or "anybody" in title or "anyone" in title or "reddit" in title or "reddit" in c or "u/" in c or "1)" in c or "weekly" in title or "daily" in title:
				print("Removing post because of bad char or link present")
				posts.pop(-1)
			elif "sunday" in ss or "monday" in title or "tuesday" in title or "wednesday" in title or "thursday" in title or "friday" in title or "saturday" in title:
				print("Removing post because of bad char or link present")
				posts.pop(-1)
		except:
			pass
	try:
		json.dump(posts, open(f"{date.today().strftime('%y_%m_%d')}{sub}_posts.json", "w"), ensure_ascii=False) 
	except:
		print("If you are here, may god help you.")
		pass
	return posts

def series(comments):
	c = []
	for i in range(len(comments)):
		for j in range(i+1, i+3):
			try:
				s = " ".join(comments[i:j+1])
				if not checkpunct(s) or counturls(s) > 0 or "u/" in s:
					continue
				c.append(s)
			except:
				pass
	return c


def getComments(dict_to_parse):
	global nlp
	# Using a dict is not perfect - if two replies are exactly the same (which is possibly very likely depending on the type of post), then their responses will be overwritten
	response_d = defaultdict(list)
	comments = []
	regex_to_remove = ["\([0-9]*[MmFf]*\)", "[0-9]+[a-zA-Z]+"]
	for p in dict_to_parse["comments"]:
		# multi-sentence comments vs single
		#comments.append(dict_to_parse["comments"][p]["text"])
		text_sentences = nlp(redditcleaner.clean(dict_to_parse["comments"][p]["text"]))
		c = [str(x) for x in text_sentences.sents if checkpunct(str(x)) and counturls(str(x)) == 0 and "u/" not in str(x)]
		for i, comment in enumerate(c):
			for reg in regex_to_remove:
				c[i] = re.sub(reg, '', c[i])
				c[i] = re.sub("  ", ' ', c[i])
		comments += c
		series_comments = series(c)
		response_keys = list(set(series_comments + c))
		comments += series_comments
		getReplies(dict_to_parse["comments"]
							   [p], response_keys, response_d)
		comments = list(set(comments))
	try:
		comments.remove("[deleted]")
	except:
		pass
	try:
		comments.remove('')
	except:
		print("No empty comment to remove in comment list for this post.")
	for i in range(len(comments)):
		if comments[i][-1] not in [".", "!", "?"]:
			comments[i] += "."
	return (comments, response_d)


def getReplies(dict_to_parse, response_keys, response_d):
	replies = []
	regex_to_remove = ["\([0-9]*[MmFf]*\)", "[0-9]+[a-zA-Z]+"]
	try:
		for p in dict_to_parse["replies"]:
			rep_opts = []
			if "text" in dict_to_parse["replies"][p]:
				text_sentences = nlp(redditcleaner.clean(dict_to_parse["replies"][p]["text"]))
				rep_opts = [str(x) for x in text_sentences.sents if checkpunct(str(x)) and counturls(str(x)) == 0]
				for i, rep_opt in enumerate(rep_opts):
					for reg in regex_to_remove:
						rep_opts[i] = re.sub(reg, '', rep_opts[i])
						rep_opts[i] = re.sub("  ", ' ', rep_opts[i])
				replies += rep_opts
				for k in response_keys:
					response_d[k] = rep_opts
			replies += getReplies(dict_to_parse["replies"][p], rep_opts, response_d)
	except:
		pass
	return replies
