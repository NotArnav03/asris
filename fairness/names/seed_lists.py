"""
FAIMR — Build-time seed lists for the name classifier corpus.

This module exists SOLELY to feed data/names/build_corpus.py.  It is
NOT imported by the audit-time runtime — the calibrated classifier
(model.pkl) plus the surname denylist (surnames.csv) replace any
need for these lists at audit time.

Why keep them at all?  Two reasons:

  1. **Multi-source agreement in training.**  Tokens that appear in
     the upstream firstname-database AND in the curated FAIMR seed
     receive a higher training weight (see build_corpus.py's merge
     function).  That weight is what makes the lookup fast-path
     correct on short ambiguous tokens like "wei" and "lee".

  2. **Bootstrappable.**  If the upstream firstname-database becomes
     unreachable, these in-repo lists are enough to train a
     usable-if-degraded model.

Anyone editing this file should run ``python data/names/build_corpus.py``
afterwards to regenerate training_corpus.csv, then
``python fairness/names/train_classifier.py`` to retrain the model.
The import-time invariant below catches the common edit error
(a token simultaneously in two sets).
"""

from __future__ import annotations


# --- Name-based gender proxies (common gendered first names) ---------------
# Organised by cultural cluster for transparency.
# Sources: US Social Security Administration top-1000 lists,
#          common South Asian, East Asian, and Arab given names.
GENDERED_NAMES: dict[str, set[str]] = {
    "male": {
        # Western English
        "james", "john", "robert", "michael", "william", "david",
        "richard", "joseph", "thomas", "charles", "daniel", "matthew",
        "anthony", "mark", "donald", "steven", "paul", "andrew",
        "kenneth", "george", "joshua", "kevin", "brian", "edward",
        "ronald", "timothy", "jason", "jeffrey", "ryan", "gary",
        "jacob", "nicholas", "eric", "jonathan", "stephen", "larry",
        "justin", "scott", "brandon", "benjamin", "samuel", "patrick",
        "frank", "raymond", "gregory", "jack", "dennis", "jerry",
        "tyler", "aaron", "adam", "henry", "nathan", "douglas",
        "zachary", "peter", "kyle", "walter", "ethan", "jeremy",
        "harold", "terry", "sean", "arthur", "christian", "austin",
        "bruce", "ralph", "roy", "noah", "russell", "alan", "philip",
        "todd", "carl", "cameron", "logan", "hunter", "mason", "liam",
        "oliver", "elijah", "lucas", "aiden", "owen", "caleb",
        "connor", "wyatt", "jayden", "gabriel", "dylan", "jordan",
        # "lee" removed — predominantly used as a surname and as a
        # unisex given name; see _UNISEX_NAMES.
        "bryan", "billy", "marcus", "christopher", "alexander",
        "sebastian", "leo", "julian", "evan", "isaac", "dominic",
        "parker", "cooper", "lincoln", "xavier", "eli", "colton",
        "nolan", "jaxon", "hudson", "levi", "landon", "jackson",
        "carson", "jameson", "grayson", "maverick", "roman", "bryson",
        "ivan", "victor", "felix", "max", "charlie", "theo", "harry",
        "oscar", "george", "freddie", "alfie", "archie", "reuben",
        # South Asian (male)
        "rahul", "amit", "vikram", "arun", "suresh", "rajesh",
        "arjun", "ravi", "sanjay", "deepak", "manish", "ajay",
        "akash", "anand", "aniket", "ankur", "aditya", "abhishek",
        "ashish", "atul", "gaurav", "harsh", "kunal", "mayank",
        "mohit", "nikhil", "nishant", "piyush", "pratik", "prateek",
        "rohit", "sachin", "sahil", "shubham", "siddharth", "sumit",
        "vaibhav", "vivek", "yash", "karan", "rohan", "sandeep",
        "vikas", "aarav", "dev", "harish", "krishna", "vishnu",
        "santosh", "ramesh", "naresh", "mahesh", "dinesh", "ganesh",
        # East Asian (male)
        # NOTE: Chinese family names (chen, li, wang, zhang, liu) and
        # the unisex Korean syllable "hyun" were removed from this list.
        # Family names carry no given-name gender signal, and including
        # them caused every East Asian candidate to be misclassified
        # male regardless of actual gender; "hyun" appeared in BOTH the
        # male and female lists, which silently cancelled to "unknown".
        # See _UNISEX_NAMES for unisex Korean/Chinese tokens.
        "wei", "ming", "jun", "yang", "xiao", "lei", "fang", "hao",
        "long", "tao", "ping", "bo", "zhen", "jian", "hiro",
        "kenji", "takashi", "naoki", "daisuke", "ryo", "yuto",
        "seung", "jae", "sung", "dong", "tae",
        # Arab / Middle Eastern (male)
        "mohammed", "omar", "hassan", "ali", "ahmed", "khalid",
        "yusuf", "ibrahim", "mustafa", "tariq", "walid", "bilal",
        "kareem", "faris", "zaid", "nabil", "rami", "samir",
        "karim", "jamal", "nasser",
    },
    "female": {
        # Western English
        "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
        "susan", "jessica", "sarah", "karen", "nancy", "lisa",
        "margaret", "betty", "sandra", "ashley", "dorothy", "kimberly",
        "emily", "donna", "michelle", "carol", "amanda", "melissa",
        "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
        "kathleen", "amy", "angela", "shirley", "anna", "brenda",
        "pamela", "emma", "nicole", "helen", "samantha", "katherine",
        "christine", "debra", "rachel", "carolyn", "janet", "catherine",
        "maria", "heather", "diane", "julie", "joyce", "victoria",
        "kelly", "christina", "lauren", "joan", "evelyn", "olivia",
        "judith", "megan", "cheryl", "martha", "andrea", "frances",
        "hannah", "jacqueline", "ann", "gloria", "teresa", "kathryn",
        "sara", "janice", "jean", "alice", "julia", "grace", "judy",
        "theresa", "rose", "beverly", "denise", "amber", "marilyn",
        "danielle", "crystal", "brittany", "natalie", "sophia",
        "madison", "isabella", "aria", "scarlett", "zoe", "chloe",
        "hazel", "lily", "mia", "ellie", "avery", "ella", "abigail",
        "aaliyah", "nora", "charlotte", "amelia", "ava", "harper",
        "luna", "camila", "sofia", "gianna", "violet", "aurora",
        "savannah", "audrey", "brooklyn", "bella", "claire", "skylar",
        "lucy", "paisley", "everly", "caroline", "nova", "emilia",
        "kennedy", "maya", "willow", "kinsley", "naomi", "elena",
        "ariel", "leah", "stella", "zara", "eva", "ivy", "ruby",
        "poppy", "daisy", "freya", "isla", "florence", "imogen",
        # South Asian (female)
        "priya", "anita", "sunita", "kavita", "neha", "pooja",
        "divya", "meena", "rekha", "anjali", "deepa", "geeta",
        "jyoti", "kritika", "lakshmi", "manisha", "nisha", "poonam",
        "radha", "rani", "shalini", "shruti", "swati", "tanvi",
        "uma", "vandana", "vineeta", "rashmi", "preeti", "pallavi",
        "namrata", "mamta", "komal", "kiran", "isha", "chandni",
        "archana", "aparna", "shreya", "riya", "tanya", "sangita",
        "namita", "sarita", "bharati",
        # East Asian (female)
        # NOTE: the Korean syllables hyun / young / min / ji / soo were
        # removed because they are routinely used across genders in
        # modern Korean given names (and "hyun" was simultaneously in
        # the male list, producing a silent cancellation).  They now
        # live in _UNISEX_NAMES and contribute no gender signal.
        "mei", "ling", "xiu", "yan", "fei", "qian", "jing", "yun",
        "shu", "xia", "akiko", "yoko", "haruko", "noriko", "keiko",
        "sachiko", "tomoko", "yuki", "sakura", "hana", "aiko",
        "eun", "na",
        "hua", "hong", "qing",
        # Arab / Middle Eastern (female)
        "fatima", "amira", "nadia", "layla", "yasmin", "nour",
        "rania", "zainab", "mariam", "hana", "dina", "lina",
        "rana", "mona", "huda", "asmaa", "salma", "aisha",
        "maryam", "sara",
    },
}


# --- Unisex given names ----------------------------------------------------
# Tokens that are statistically used across genders in their source culture.
# These are MATCHED at corpus-build time so build_corpus.py marks them
# with p_female=0.5 and weight=0.5, contributing as unisex signal during
# classifier training.
#
# Curated conservatively — every token here was either (a) present in both
# the male and female lists in a prior revision (a structural bug) or
# (b) routinely used across genders in the source culture per public
# naming statistics.
_UNISEX_NAMES: set = {
    # Korean syllables that appear as unisex given names.
    # ("eun" is intentionally LEFT in the female list — it is strongly
    # female-coded in modern Korean usage despite occasional male use,
    # and the import-time invariant guards against re-adding it here.)
    "hyun", "young", "min", "ji", "soo", "jin", "joon", "hye",
    # Common Chinese given-name characters used across genders
    "yu", "an",
    # Western unisex (most-flagrant cases)
    "lee",
}


# --- Vocab consistency invariant ------------------------------------------
# Fail fast at import time if the seed lists develop cross-list
# contamination.  These invariants are LOAD-BEARING for the training
# corpus: a single token appearing in two sets would produce ambiguous
# labels that contaminate the classifier's view of similar-looking
# names.  Stated as runtime asserts so a careless edit cannot ship
# silently.
def _assert_name_vocab_invariants() -> None:
    male = GENDERED_NAMES["male"]
    female = GENDERED_NAMES["female"]
    overlap_mf = male & female
    overlap_mu = male & _UNISEX_NAMES
    overlap_fu = female & _UNISEX_NAMES
    if overlap_mf:
        raise AssertionError(
            f"GENDERED_NAMES: male/female collision: {sorted(overlap_mf)}"
        )
    if overlap_mu:
        raise AssertionError(
            f"GENDERED_NAMES: male/unisex collision: {sorted(overlap_mu)}"
        )
    if overlap_fu:
        raise AssertionError(
            f"GENDERED_NAMES: female/unisex collision: {sorted(overlap_fu)}"
        )


_assert_name_vocab_invariants()
