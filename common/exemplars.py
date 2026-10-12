"""Hand-written 3-shot in-context exemplars (one block per target language).

INDIC_MMT_PROMPTS_3 is used verbatim for the "+ TDV" few-shot strategies.
INDIC_MMT_PROMPTS_3_NO_TDV is the same exemplars with every line that mentions
the video description removed; it is used for the few-shot-only strategy so
that no video grounding leaks into a text-only prompt.
"""

import re

INDIC_MMT_PROMPTS_3 = {
    "hindi": """
    If the English sentence is ambiguous, use the video description to choose the correct meaning in your translation.
    # Examples:
    # 1. English: "He is playing bass."
    #    Video Description: "A man is playing a bass guitar on stage."
    #    Translation (Hindi): "वह बास गिटार बजा रहा है।" (Musical instrument)
    # 2. English: "The crane is flying."
    #    Video Description: "A large bird is soaring above a lake."
    #    Translation (Hindi): "सारस उड़ रहा है।" (Bird)
    # 3. English: "The bank is closed."
    #    Video Description: "A person walking along the edge of a river."
    #    Translation (Hindi): "नदी का किनारा बंद है।" (River bank vs Financial)
    """,

    "bengali": """
    If the English sentence is ambiguous, use the video description to choose the correct Bengali translation.
    # Examples:
    # 1. English: "The light is very low."
    #    Video Description: "A person carrying a suitcase that weighs very little."
    #    Translation (Bengali): "জিনিসটি খুব হালকা।" (Weight)
    # 2. English: "He is sitting by the bank."
    #    Video Description: "A man sitting on the edge of a river."
    #    Translation (Bengali): "সে নদীর পাড়ে বসে আছে।" (River bank)
    # 3. English: "I need a new mouse."
    #    Video Description: "A person using a computer in an office."
    #    Translation (Bengali): "আমার একটি নতুন মাউস দরকার।" (Computer peripheral)
    """,

    "tamil": """
    Use the video description to resolve English ambiguities for Tamil translation.
    # Examples:
    # 1. English: "The match is starting."
    #    Video Description: "A person strikes a wooden stick to light a fire."
    #    Translation (Tamil): "தீக்குச்சி எரியத் தொடங்குகிறது।" (Matchstick)
    # 2. English: "He saw a star."
    #    Video Description: "A famous actor walking down the red carpet."
    #    Translation (Tamil): "அவர் ஒரு பிரபலமான நடிகரைப் பார்த்தார்।" (Celebrity vs Celestial body)
    # 3. English: "The plant is growing."
    #    Video Description: "Smoke coming out of a large chemical factory."
    #    Translation (Tamil): "அந்தத் தொழிற்சாலை வளர்ந்து வருகிறது।" (Factory vs Flora)
    """,

    "telugu": """
    Use the video description to resolve English ambiguities for Telugu translation.
    # Examples:
    # 1. English: "Look at that bow."
    #    Video Description: "An archer holds a weapon to shoot an arrow."
    #    Translation (Telugu): "ఆ విల్లును చూడండి।" (Weapon)
    # 2. English: "The spring is beautiful."
    #    Video Description: "Water bubbling up from a natural underground source."
    #    Translation (Telugu): "ఆ ఊట చాలా అందంగా ఉంది।" (Water source vs Season)
    # 3. English: "He left the club."
    #    Video Description: "A person puts down a heavy wooden bat used for exercise."
    #    Translation (Telugu): "అతను ఆ గదని వదిలేశాడు।" (Heavy stick/Gada vs Social club)
    """,

    "marathi": """
    Use the video description to determine the correct Marathi noun or verb form.
    # Examples:
    # 1. English: "He is playing with a bat."
    #    Video Description: "A nocturnal mammal is flying in a cave."
    #    Translation (Marathi): "तो वटवाघळासोबत खेळत आहे।" (Animal)
    # 2. English: "The watch is broken."
    #    Video Description: "A security guard looking at a monitor."
    #    Translation (Marathi): "पहारा तुटला आहे।" (Duty/Guard vs Timepiece)
    # 3. English: "Give me the file."
    #    Video Description: "A carpenter smoothing a piece of wood with a metal tool."
    #    Translation (Marathi): "मला कानस दे।" (Metal tool vs Document)
    """,

    "odia": """
    Use the video description to resolve English ambiguities for Odia translation.
    # Examples:
    # 1. English: "The bark is rough."
    #    Video Description: "Close up of the outer layer of a tree trunk."
    #    Translation (Odia): "ଗଛର ଛାଲି ଖସଖସିଆ ଅଟେ।" (Tree part)
    # 2. English: "He is at the bar."
    #    Video Description: "A gymnast hanging from a horizontal metal pole."
    #    Translation (Odia): "ସେ ସେହି ଲୁହା ବାର୍‌ରେ ଝୁଲୁଛନ୍ତି।" (Gymnastics pole)
    # 3. English: "The scale is small."
    #    Video Description: "A close up of the skin of a fish."
    #    Translation (Odia): "ମାଛର କାତି ଛୋଟ ଅଟେ।" (Fish skin vs Weighing machine)
    """,

    "assamese": """
    Use the video description to resolve English ambiguities for Assamese translation.
    # Examples:
    # 1. English: "Put it in the trunk."
    #    Video Description: "A man puts luggage in the back of a car."
    #    Translation (Assamese): "এইটো গাড়ীৰ পিছফালে থোৱা।" (Car boot)
    # 2. English: "The fan is loud."
    #    Video Description: "A group of people screaming for a singer."
    #    Translation (Assamese): "অনুৰাগীসকলে বৰকৈ চিঞৰিছে।" (Follower)
    # 3. English: "The glass is full."
    #    Video Description: "A close up of a window pane."
    #    Translation (Assamese): "খিড়িকীৰ আইনাখন সম্পূৰ্ণ।" (Window glass vs Drinking cup)
    """,

    "punjabi": """
    Use the video description to resolve English ambiguities for Punjabi translation.
    # Examples:
    # 1. English: "The fan is loud."
    #    Video Description: "A crowd of people cheering for a celebrity."
    #    Translation (Punjabi): "ਪ੍ਰਸ਼ੰਸਕ ਬਹੁਤ ਰੌਲਾ ਪਾ ਰਹੇ ਹਨ।" (Follower)
    # 2. English: "He is using a pen."
    #    Video Description: "A farmer putting cattle into a fenced area."
    #    Translation (Punjabi): "ਉਹ ਪਸ਼ੂਆਂ ਦੇ ਵਾੜੇ ਦੀ ਵਰਤੋਂ ਕਰ ਰਿਹਾ ਹੈ।" (Animal enclosure vs Writing tool)
    # 3. English: "The mole is dark."
    #    Video Description: "A small brown spot on a person's skin."
    #    Translation (Punjabi): "ਤਿਲ ਦਾ ਰੰਗ ਗੂੜ੍ਹਾ ਹੈ।" (Skin mark vs Animal)
    """,

    "kannada": """
    Use the video description to resolve English ambiguities for Kannada translation.
    # Examples:
    # 1. English: "The bill is ready."
    #    Video Description: "A bird's beak is shown in a close-up shot."
    #    Translation (Kannada): "ಪಕ್ಷಿಯ ಕೊಕ್ಕು ಸಿದ್ಧವಾಗಿದೆ।" (Beak)
    # 2. English: "He felt the rose."
    #    Video Description: "A person stood up from their chair."
    #    Translation (Kannada): "ಅವನು ಎದ್ದು ನಿಂತನು।" (Past tense of 'rise' vs Flower)
    # 3. English: "The lead is heavy."
    #    Video Description: "A scientist holding a block of grey metal."
    #    Translation (Kannada): "ಸೀಸವು ತುಂಬಾ ಭಾರವಾಗಿದೆ।" (Metal vs Guide)
    """,

    "gujarati": """
    Use the video description to resolve English ambiguities for Gujarati translation.
    # Examples:
    # 1. English: "He is at the bar."
    #    Video Description: "A gymnast performing on a horizontal metal pole."
    #    Translation (Gujarati): "તે બાર (પોલ) પર છે।" (Gymnastics pole)
    # 2. English: "The letter is long."
    #    Video Description: "An image showing a very long character of the alphabet."
    #    Translation (Gujarati): "આ અક્ષર ઘણો લાંબો છે।" (Character/Alphabet vs Mail)
    # 3. English: "The suit is black."
    #    Video Description: "A lawyer wearing a formal jacket and trousers."
    #    Translation (Gujarati): "તેમનો પોશાક કાળો છે।" (Clothing vs Lawsuit)
    """,

    "malayalam": """
    Use the video description to resolve English ambiguities for Malayalam translation.
    # Examples:
    # 1. English: "The mouse is moving."
    #    Video Description: "A small rodent running across the kitchen floor."
    #    Translation (Malayalam): "എലി ഓടുകയാണ്।" (Animal)
    # 2. English: "He has a lot of soul."
    #    Video Description: "A person looking at the bottom of their shoe."
    #    Translation (Malayalam): "അയാളുടെ ഷൂവിന്റെ അടിഭാഗം ശ്രദ്ധിക്കുക।" (Sole of shoe vs Spirit)
    # 3. English: "The rock is hard."
    #    Video Description: "A guitarist playing heavy metal music."
    #    Translation (Malayalam): "റോക്ക് സംഗീതം കഠിനമാണ്।" (Music genre vs Stone)
    """
}

def _strip_video_desc(text):
    lines = [l for l in text.split("\n") if "video description" not in l.lower()]
    return re.sub(r"\n\s*\n+", "\n", "\n".join(lines)).strip("\n")


INDIC_MMT_PROMPTS_3_NO_TDV = {
    lang: _strip_video_desc(text) for lang, text in INDIC_MMT_PROMPTS_3.items()
}
