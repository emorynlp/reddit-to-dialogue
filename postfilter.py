import os, json, copy, spacy, time, re


regex_to_remove = ["\([0-9]*[MmFf]*\)", "\([MmFf]*[0-9]*\)", "[0-9]+[a-zA-Z]+", "[[a-zA-Z]+[0-9]+]"]

nlp = spacy.load('en_core_web_sm')


length = 50
total = 0.0
filt = 0.0

def postfilter(sub):
    for path, subdirs, files in os.walk("output"):
        for name in files:
            if ".json" not in name or sub not in name: continue
            a = json.load(open(os.path.join(path, name), "r"))
            sub = list(a.keys())[0]
            b = {}
            b[sub] = []
            for i in list(a.values())[0]:
                
                t = False
                s = set()
                for string in i["dialogue"]:
                    if string in s:
                        t = True
                        break
                    text_sentences = nlp(string)
                    for split in text_sentences.sents:
                        if str(split) in s:
                            t = True
                            #print(str(split), name)
                            break
                        s.add(str(split))
                    if t: break
                total+=1
                filt+=1
                if not t:
                    filt-=1
                    b[sub].append(copy.copy(i)) 
                    for idx in range(len(b[sub][-1]["dialogue"])):
                        for reg in regex_to_remove:
                            b[sub][-1]["dialogue"][idx] = re.sub(reg, '', b[sub][-1]["dialogue"][idx])
                        b[sub][-1]["dialogue"][idx] = re.sub("  ", ' ', b[sub][-1]["dialogue"][idx])
                
                print(f"{filt/total*100:.2f}%, {total}")
            json.dump(b, open("postfiltered_"+name, "w"), indent=2, ensure_ascii=False)
                    
                    
    print(f"{filt}, {total}")
