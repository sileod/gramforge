def negate_predicate(predicate):
    replacements = {
        "is ": "is not ",
        "has ": "does not have ",
        "does ": "does not ",
        'can ': 'cannot '
    }
    for key, val in replacements.items():
        if predicate.startswith(key):
            return predicate.replace(key, val, 1)
    
    words = predicate.split(" ")
    if words[0].endswith("s"):
        words[0] = words[0][:-1]  # Remove 's' from the verb
    return "does not " + " ".join(words)



short_propositions = [
    "Planet Xylos has diamond rain.",
    "Bellbridge's houses are all purple.",
    "Gravity inverts in Oakhaven on Tuesdays.",
    "A tree in Whispering Woods has golden fruit.",
    "A square cloud is over Silver Lake.",
    "Blackwood Manor's mirrors show the future.",
    "Avani's people eat only blue moss.",
    "A singing flower blooms in the Amazon.",
    "John Smith's car runs on ethanol.",
    "Eldoria's children talk to animals."
]




neg_short_propositions = [
    "Planet Xylos has no diamond rain.",
    "Not all Bellbridge's houses are purple.",
    "Gravity does not invert in Oakhaven on Tuesdays.",
    "No tree in Whispering Woods has golden fruit.",
    "No square cloud is over Silver Lake.",
    "Blackwood Manor's mirrors do not show the future.",
    "Avani's people do not eat only blue moss.",
    "No singing flower blooms in the Amazon.",
    "John Smith's car does not run on ethanol.",
    "Eldoria's children do not talk to animals."
]
 
#are any of those logically entailing or contradicting each other in a clear-cut way?
# I'm not talking about slight contrast, correlation or negataive correlation

predicates = [
    "is a client of Meta", "is a client of Costco", "is a client of LVMH", "uses an ios phone",
    "owns an Android phone", "owns a smart tv", "watches fantasy movies",
    "reads mystery novels", "writes a travel blog", "practices digital art",
    "enjoys virtual reality gaming", "is a drone photographer", "enjoys making ceramics", "enjoys watercolor painting", "practices graffiti art",
    "writes poetry", "practices calligraphy", "enjoys landscape photography", "enjoys macrame", "enjoys origami",
    "can play the piano", "can play the guitar",
    "can play the harmonica", "can play the flute", "plays the drums", "plays the violin", "enjoys salsa dancing", "does enjoy trail running",
    "does enjoy mountain biking", "enjoys snowboarding", "enjoys spelunking", "enjoys cross-country skiing",
    "enjoys stand-up paddleboarding", "enjoys windsurfing", "practices pilates",
    "practices tai chi", "practices zumba", "practices archery", "practices kickboxing",
    "enjoys skydiving", "collects foreign coins", "collects vintage stamps", "collects modern art",
    "collects vintage vinyl records", "collects action figures", "collects antique clocks",
    "collect rare sneakers", "collects antique jewelry", "collects luxury watches", "collects first edition books",
    "collects classic novels", "collects comic books", "enjoys fishing", "enjoys stargazing",
    "practices urban gardening", "enjoys rooftop gardening", "owns a microscope",
    "works on fridays", "drives a hybrid car", "has a pet dog", "is right-handed", "is a night owl", "uses contact lenses",
    "has a tattoo", "has a piercing", "travels domestically frequently", "is allergic to anything",
    "has lived in exactly three countries", "knows morse code", "makes homemade flans", "bakes bread at home",
    "is a tea enthusiast", "is a coffee connoisseur", "is a scotch connoisseur", "is a craft beer aficionado", "has a saltwater aquarium", "builds model airplanes", "owns a very old television",
    "enjoys logic puzzles", "uses a Windows laptop", "is a Linux enthusiast", "is a cybersecurity expert",
    "enjoys coding in Python", "streams on Twitch", "owns a smartwatch", "owns a 3D printer",
    "plays eSports competitively", "develops open-source software projects in their free time", "frequently participates in hackathons and coding competitions",
    "owns a high-end gaming PC with custom-built components", "regularly contributes to tech forums and online communities",
    "is an active member of a local robotics club", "creates augmented reality experiences for mobile applications",
    "works as a freelance web developer specializing in e-commerce sites", "hosts a popular podcast about emerging technologies",
    "maintains a personal blog focused on cybersecurity tips", "is a dedicated advocate for digital privacy and encryption",
    "creates large-scale murals for public art installations", "writes and illustrates their own graphic novels",
    "composes and performs experimental electronic music", "hosts regular workshops on creative writing",
    "is a member of a local theater group specializing in improv", "designs and sews custom cosplay costumes for conventions",
    "makes intricate hand-cut paper art for exhibitions", "hosts a YouTube channel dedicated to art tutorials",
    "creates bespoke furniture pieces from reclaimed wood", "is a professional photographer specializing in portrait photography",
    "trains for and competes in international triathlons", "is a certified yoga instructor teaching classes weekly",
    "plays as a goalkeeper for a local amateur soccer team", "participates in long-distance cycling events across the country",
    "is an avid mountain climber who has scaled several peaks", "mentors a youth basketball team on weekends",
    "competes in national level swimming championships", "practices and performs acrobatic dance routines",
    "enjoys deep-sea diving and exploring underwater caves",
    "collects rare and antique scientific instruments", "has a vast collection of first-edition science fiction novels",
    "owns an extensive assortment of vintage comic book memorabilia", "is passionate about collecting and restoring classic cars",
    "collects limited-edition art prints from contemporary artists", "has a curated collection of mid-century modern furniture",
    "collects historical artifacts related to ancient civilizations", "owns a significant collection of rare gemstones and minerals",
    "is an avid collector of autographed memorabilia from famous musicians", "has a specialized collection of handmade artisan pottery",
    "is an enthusiastic bird watcher who travels for rare sightings", "maintains a large, organic vegetable garden year-round",
    "volunteers for local environmental conservation projects", "regularly goes on multi-day backpacking trips in national parks",
    "enjoys kayaking and exploring remote waterways", "is a certified scuba diver with advanced underwater photography skills",
    "participates in citizen science projects related to wildlife monitoring", "is an expert in identifying and foraging wild edible plants",
    "enjoys camping and organizing outdoor survival workshops", "is dedicated to sustainable living and zero-waste practices",
    "is a culinary enthusiast who experiments with international cuisines", "hosts themed dinner parties featuring gourmet home-cooked meals",
    "is a wine connoisseur with a private cellar of vintage wines", "is a dedicated volunteer for local community service projects",
    "enjoys writing detailed reviews of new and classic board games", "is a chess master who participates in national tournaments",
    "hosts regular game nights featuring complex strategy games", "is an amateur astronomer who builds and uses custom telescopes",
    "writes in-depth travel guides for off-the-beaten-path destinations"
]
