"""Confirmed 26-man Qatar 2022 squads, matched to FIFA 23 for a leak-free backtest."""

from __future__ import annotations

import warnings

import pandas as pd
from rapidfuzz import fuzz, process

from . import config as C
from .data import load_fifa_players
from .features import edition_table_filled, _STRENGTH
from .squads import _norm, _strength_from_players

warnings.filterwarnings("ignore")

# FIFA 23 (released Sept 2022) is the edition active before the Nov 2022 World Cup.
EDITION = 2023
MIN_MATCHED = 16          # below this a team falls back to the nationality-pool proxy
FUZZY_CUTOFF = 86

# FIFA short-name nicknames the fuzzy matcher would otherwise miss.
_ALIAS = {"vinicius junior": "vini jr", "son heung-min": "son heung min"}

# Best-effort reconstruction of the 32 confirmed squads (G/D/M/F position groups).
SQUADS_2022: dict[str, dict[str, list[str]]] = {
    "Qatar": {
        "G": ["Saad Al Sheeb", "Meshaal Barsham", "Yousef Hassan"],
        "D": ["Pedro Miguel", "Bassam Al-Rawi", "Boualem Khoukhi", "Tarek Salman",
              "Homam Ahmed", "Abdelkarim Hassan", "Musab Kheder", "Jassem Gaber"],
        "M": ["Karim Boudiaf", "Assim Madibo", "Salem Al-Hajri", "Mohammed Waad",
              "Abdulaziz Hatem", "Ali Assadalla", "Ahmed Fathi"],
        "F": ["Hassan Al-Haydos", "Akram Afif", "Almoez Ali", "Mohammed Muntari",
              "Khalid Muneer", "Ismaeel Mohammad", "Naif Al-Hadhrami"],
    },
    "Ecuador": {
        "G": ["Hernan Galindez", "Alexander Dominguez", "Moises Ramirez"],
        "D": ["Piero Hincapie", "Felix Torres", "Robert Arboleda", "Pervis Estupinan",
              "Angelo Preciado", "Diego Palacios", "Jackson Porozo", "William Pacho"],
        "M": ["Moises Caicedo", "Carlos Gruezo", "Jose Cifuentes", "Jhegson Mendez",
              "Romario Ibarra", "Angel Mena", "Alan Franco", "Jeremy Sarmiento"],
        "F": ["Enner Valencia", "Michael Estrada", "Gonzalo Plata", "Kevin Rodriguez",
              "Djorkaeff Reasco"],
    },
    "Senegal": {
        "G": ["Edouard Mendy", "Alfred Gomis", "Seny Dieng"],
        "D": ["Kalidou Koulibaly", "Abdou Diallo", "Youssouf Sabaly", "Saliou Ciss",
              "Fode Ballo-Toure", "Pape Abou Cisse", "Formose Mendy"],
        "M": ["Idrissa Gueye", "Cheikhou Kouyate", "Nampalys Mendy", "Pape Matar Sarr",
              "Pathe Ciss", "Krepin Diatta", "Pape Gueye"],
        "F": ["Sadio Mane", "Ismaila Sarr", "Boulaye Dia", "Famara Diedhiou",
              "Bamba Dieng", "Iliman Ndiaye", "Nicolas Jackson"],
    },
    "Netherlands": {
        "G": ["Andries Noppert", "Remko Pasveer", "Justin Bijlow"],
        "D": ["Virgil van Dijk", "Matthijs de Ligt", "Nathan Ake", "Stefan de Vrij",
              "Jurrien Timber", "Daley Blind", "Denzel Dumfries", "Tyrell Malacia"],
        "M": ["Frenkie de Jong", "Steven Berghuis", "Marten de Roon", "Teun Koopmeiners",
              "Davy Klaassen", "Kenneth Taylor", "Xavi Simons"],
        "F": ["Memphis Depay", "Steven Bergwijn", "Cody Gakpo", "Vincent Janssen",
              "Luuk de Jong", "Noa Lang", "Wout Weghorst"],
    },
    "England": {
        "G": ["Jordan Pickford", "Aaron Ramsdale", "Nick Pope"],
        "D": ["Harry Maguire", "John Stones", "Kyle Walker", "Luke Shaw",
              "Kieran Trippier", "Trent Alexander-Arnold", "Eric Dier", "Ben White", "Conor Coady"],
        "M": ["Declan Rice", "Jordan Henderson", "Jude Bellingham", "Mason Mount",
              "Kalvin Phillips", "Conor Gallagher"],
        "F": ["Harry Kane", "Raheem Sterling", "Bukayo Saka", "Marcus Rashford",
              "Phil Foden", "Jack Grealish", "Callum Wilson", "James Maddison"],
    },
    "Iran": {
        "G": ["Alireza Beiranvand", "Amir Abedzadeh", "Hossein Hosseini"],
        "D": ["Morteza Pouraliganji", "Hossein Kanaani", "Shojae Khalilzadeh", "Sadegh Moharrami",
              "Ehsan Hajsafi", "Milad Mohammadi", "Ramin Rezaeian", "Majid Hosseini", "Abolfazl Jalali"],
        "M": ["Saeid Ezatolahi", "Ahmad Nourollahi", "Vahid Amiri", "Alireza Jahanbakhsh",
              "Saman Ghoddos", "Ali Karimi", "Mehdi Torabi", "Ali Gholizadeh"],
        "F": ["Sardar Azmoun", "Mehdi Taremi", "Karim Ansarifard", "Allahyar Sayyadmanesh"],
    },
    "United States": {
        "G": ["Matt Turner", "Sean Johnson", "Ethan Horvath"],
        "D": ["Walker Zimmerman", "Tim Ream", "Aaron Long", "Cameron Carter-Vickers",
              "Sergino Dest", "Antonee Robinson", "DeAndre Yedlin", "Joe Scally", "Shaq Moore"],
        "M": ["Tyler Adams", "Weston McKennie", "Yunus Musah", "Kellyn Acosta",
              "Luca de la Torre", "Cristian Roldan"],
        "F": ["Christian Pulisic", "Tim Weah", "Brenden Aaronson", "Gio Reyna",
              "Jesus Ferreira", "Jordan Morris", "Haji Wright", "Josh Sargent"],
    },
    "Wales": {
        "G": ["Wayne Hennessey", "Danny Ward", "Adam Davies"],
        "D": ["Ben Davies", "Joe Rodon", "Chris Mepham", "Connor Roberts",
              "Neco Williams", "Ethan Ampadu", "Tom Lockyer", "Chris Gunter"],
        "M": ["Aaron Ramsey", "Joe Allen", "Harry Wilson", "Joe Morrell",
              "Jonny Williams", "Dylan Levitt", "Matthew Smith", "Sorba Thomas", "Rubin Colwill"],
        "F": ["Gareth Bale", "Daniel James", "Kieffer Moore", "Brennan Johnson", "Mark Harris"],
    },
    "Argentina": {
        "G": ["Emiliano Martinez", "Franco Armani", "Geronimo Rulli"],
        "D": ["Nahuel Molina", "Gonzalo Montiel", "Cristian Romero", "German Pezzella",
              "Nicolas Otamendi", "Lisandro Martinez", "Marcos Acuna", "Nicolas Tagliafico", "Juan Foyth"],
        "M": ["Rodrigo De Paul", "Leandro Paredes", "Guido Rodriguez", "Alexis Mac Allister",
              "Enzo Fernandez", "Exequiel Palacios", "Alejandro Gomez"],
        "F": ["Lionel Messi", "Angel Di Maria", "Lautaro Martinez", "Julian Alvarez",
              "Nicolas Gonzalez", "Joaquin Correa", "Paulo Dybala"],
    },
    "Saudi Arabia": {
        "G": ["Mohammed Al-Owais", "Nawaf Al-Aqidi", "Mohammed Al-Rubaie"],
        "D": ["Yasser Al-Shahrani", "Saud Abdulhamid", "Ali Al-Bulaihi", "Hassan Tambakti",
              "Abdulelah Al-Amri", "Mohammed Al-Breik", "Abdullah Madu", "Sultan Al-Ghanam"],
        "M": ["Salman Al-Faraj", "Mohamed Kanno", "Abdulellah Al-Malki", "Nasser Al-Dawsari",
              "Sami Al-Najei", "Ali Al-Hassan", "Abdulrahman Al-Aboud"],
        "F": ["Salem Al-Dawsari", "Firas Al-Buraikan", "Saleh Al-Shehri", "Haitham Asiri",
              "Khalid Al-Ghannam", "Nawaf Al-Abed"],
    },
    "Mexico": {
        "G": ["Guillermo Ochoa", "Alfredo Talavera", "Rodolfo Cota"],
        "D": ["Cesar Montes", "Hector Moreno", "Nestor Araujo", "Jorge Sanchez",
              "Jesus Gallardo", "Kevin Alvarez", "Gerardo Arteaga", "Johan Vasquez"],
        "M": ["Edson Alvarez", "Hector Herrera", "Andres Guardado", "Luis Chavez",
              "Erick Gutierrez", "Carlos Rodriguez", "Roberto Alvarado", "Orbelin Pineda", "Uriel Antuna"],
        "F": ["Raul Jimenez", "Hirving Lozano", "Henry Martin", "Alexis Vega", "Rogelio Funes Mori"],
    },
    "Poland": {
        "G": ["Wojciech Szczesny", "Lukasz Skorupski", "Kamil Grabara"],
        "D": ["Kamil Glik", "Jan Bednarek", "Bartosz Bereszynski", "Mateusz Wieteska",
              "Robert Gumny", "Matty Cash", "Nicola Zalewski", "Artur Jedrzejczyk"],
        "M": ["Piotr Zielinski", "Grzegorz Krychowiak", "Krystian Bielik", "Sebastian Szymanski",
              "Damian Szymanski", "Jakub Kaminski", "Przemyslaw Frankowski", "Kamil Jozwiak"],
        "F": ["Robert Lewandowski", "Arkadiusz Milik", "Krzysztof Piatek", "Karol Swiderski"],
    },
    "France": {
        "G": ["Hugo Lloris", "Steve Mandanda", "Alphonse Areola"],
        "D": ["Raphael Varane", "Dayot Upamecano", "Ibrahima Konate", "Jules Kounde",
              "Benjamin Pavard", "Theo Hernandez", "Lucas Hernandez", "Axel Disasi", "William Saliba"],
        "M": ["Aurelien Tchouameni", "Adrien Rabiot", "Eduardo Camavinga", "Youssouf Fofana",
              "Matteo Guendouzi", "Jordan Veretout"],
        "F": ["Kylian Mbappe", "Olivier Giroud", "Antoine Griezmann", "Ousmane Dembele",
              "Marcus Thuram", "Kingsley Coman", "Randal Kolo Muani"],
    },
    "Australia": {
        "G": ["Mathew Ryan", "Andrew Redmayne", "Danny Vukovic"],
        "D": ["Harry Souttar", "Kye Rowles", "Milos Degenek", "Aziz Behich",
              "Nathaniel Atkinson", "Fran Karacic", "Bailey Wright", "Joel King", "Thomas Deng"],
        "M": ["Aaron Mooy", "Jackson Irvine", "Ajdin Hrustic", "Riley McGree",
              "Cameron Devlin", "Keanu Baccus"],
        "F": ["Mathew Leckie", "Mitchell Duke", "Craig Goodwin", "Awer Mabil",
              "Jamie Maclaren", "Jason Cummings", "Garang Kuol"],
    },
    "Denmark": {
        "G": ["Kasper Schmeichel", "Frederik Ronnow", "Oliver Christensen"],
        "D": ["Simon Kjaer", "Andreas Christensen", "Joachim Andersen", "Jannik Vestergaard",
              "Joakim Maehle", "Rasmus Kristensen", "Daniel Wass", "Victor Nelsson", "Alexander Bah"],
        "M": ["Pierre-Emile Hojbjerg", "Thomas Delaney", "Christian Eriksen", "Mathias Jensen",
              "Christian Norgaard", "Robert Skov"],
        "F": ["Andreas Skov Olsen", "Mikkel Damsgaard", "Jesper Lindstrom", "Jonas Wind",
              "Martin Braithwaite", "Kasper Dolberg", "Andreas Cornelius"],
    },
    "Tunisia": {
        "G": ["Aymen Dahmen", "Bechir Ben Said", "Mouez Hassen"],
        "D": ["Montassar Talbi", "Yassine Meriah", "Dylan Bronn", "Ali Abdi",
              "Mohamed Drager", "Wajdi Kechrida", "Nader Ghandri", "Bilel Ifa"],
        "M": ["Aissa Laidouni", "Ellyes Skhiri", "Ferjani Sassi", "Mohamed Ali Ben Romdhane",
              "Anis Ben Slimane", "Ghaylene Chaalali", "Hannibal Mejbri"],
        "F": ["Youssef Msakni", "Wahbi Khazri", "Naim Sliti", "Issam Jebali",
              "Seifeddine Jaziri", "Taha Yassine Khenissi"],
    },
    "Spain": {
        "G": ["Unai Simon", "Robert Sanchez", "David Raya"],
        "D": ["Pau Torres", "Aymeric Laporte", "Eric Garcia", "Hugo Guillamon",
              "Cesar Azpilicueta", "Jordi Alba", "Dani Carvajal", "Jose Gaya", "Alejandro Balde"],
        "M": ["Sergio Busquets", "Rodri", "Pedri", "Gavi", "Koke", "Carlos Soler", "Marcos Llorente"],
        "F": ["Alvaro Morata", "Ferran Torres", "Marco Asensio", "Dani Olmo",
              "Pablo Sarabia", "Yeremi Pino", "Ansu Fati", "Nico Williams"],
    },
    "Costa Rica": {
        "G": ["Keylor Navas", "Esteban Alvarado", "Patrick Sequeira"],
        "D": ["Oscar Duarte", "Francisco Calvo", "Kendall Waston", "Juan Pablo Vargas",
              "Keysher Fuller", "Bryan Oviedo", "Daniel Chacon", "Carlos Martinez"],
        "M": ["Yeltsin Tejeda", "Celso Borges", "Youstin Salas", "Gerson Torres",
              "Roan Wilson", "Douglas Lopez", "Brandon Aguilera"],
        "F": ["Joel Campbell", "Anthony Contreras", "Johan Venegas", "Jewison Bennette",
              "Alvaro Zamora", "Bryan Ruiz"],
    },
    "Germany": {
        "G": ["Manuel Neuer", "Marc-Andre ter Stegen", "Kevin Trapp"],
        "D": ["Antonio Rudiger", "Niklas Sule", "Thilo Kehrer", "Matthias Ginter",
              "Nico Schlotterbeck", "David Raum", "Lukas Klostermann", "Armel Bella-Kotchap", "Christian Gunter"],
        "M": ["Joshua Kimmich", "Ilkay Gundogan", "Leon Goretzka", "Jamal Musiala",
              "Kai Havertz", "Julian Brandt", "Mario Gotze", "Leroy Sane"],
        "F": ["Thomas Muller", "Serge Gnabry", "Niclas Fullkrug", "Karim Adeyemi", "Youssoufa Moukoko"],
    },
    "Japan": {
        "G": ["Shuichi Gonda", "Daniel Schmidt", "Eiji Kawashima"],
        "D": ["Maya Yoshida", "Ko Itakura", "Takehiro Tomiyasu", "Hiroki Sakai",
              "Yuto Nagatomo", "Miki Yamane", "Shogo Taniguchi", "Hiroki Ito", "Yuta Nakayama"],
        "M": ["Wataru Endo", "Hidemasa Morita", "Gaku Shibasaki", "Ao Tanaka", "Junya Ito",
              "Ritsu Doan", "Takefusa Kubo", "Kaoru Mitoma", "Daichi Kamada"],
        "F": ["Daizen Maeda", "Takuma Asano", "Ayase Ueda", "Shuto Machino"],
    },
    "Belgium": {
        "G": ["Thibaut Courtois", "Simon Mignolet", "Koen Casteels"],
        "D": ["Toby Alderweireld", "Jan Vertonghen", "Leander Dendoncker", "Wout Faes",
              "Zeno Debast", "Timothy Castagne", "Thomas Meunier", "Arthur Theate"],
        "M": ["Kevin De Bruyne", "Youri Tielemans", "Axel Witsel", "Amadou Onana",
              "Hans Vanaken", "Yannick Carrasco"],
        "F": ["Romelu Lukaku", "Eden Hazard", "Dries Mertens", "Michy Batshuayi",
              "Leandro Trossard", "Jeremy Doku", "Charles De Ketelaere", "Thorgan Hazard", "Lois Openda"],
    },
    "Canada": {
        "G": ["Milan Borjan", "Dayne St. Clair", "James Pantemis"],
        "D": ["Alistair Johnston", "Kamal Miller", "Steven Vitoria", "Sam Adekugbe",
              "Richie Laryea", "Joel Waterman", "Derek Cornelius"],
        "M": ["Stephen Eustaquio", "Atiba Hutchinson", "Jonathan Osorio", "Mark-Anthony Kaye",
              "Samuel Piette", "Ismael Kone", "David Wotherspoon", "Liam Fraser"],
        "F": ["Alphonso Davies", "Jonathan David", "Cyle Larin", "Tajon Buchanan",
              "Junior Hoilett", "Lucas Cavallini", "Ike Ugbo"],
    },
    "Morocco": {
        "G": ["Yassine Bounou", "Munir Mohamedi", "Ahmed Reda Tagnaouti"],
        "D": ["Achraf Hakimi", "Noussair Mazraoui", "Romain Saiss", "Nayef Aguerd",
              "Jawad El Yamiq", "Yahya Attiyat-Allah", "Achraf Dari", "Badr Benoun"],
        "M": ["Sofyan Amrabat", "Azzedine Ounahi", "Selim Amallah", "Abdelhamid Sabiri",
              "Bilal El Khannouss", "Yahya Jabrane"],
        "F": ["Hakim Ziyech", "Sofiane Boufal", "Youssef En-Nesyri", "Zakaria Aboukhlal",
              "Abde Ezzalzouli", "Walid Cheddira", "Ilias Chair", "Anass Zaroury"],
    },
    "Croatia": {
        "G": ["Dominik Livakovic", "Ivica Ivusic", "Ivo Grbic"],
        "D": ["Dejan Lovren", "Josip Juranovic", "Borna Barisic", "Josko Gvardiol",
              "Josip Sutalo", "Borna Sosa", "Martin Erlic", "Josip Stanisic"],
        "M": ["Luka Modric", "Mateo Kovacic", "Marcelo Brozovic", "Mario Pasalic",
              "Nikola Vlasic", "Lovro Majer", "Kristijan Jakic", "Luka Sucic"],
        "F": ["Ivan Perisic", "Andrej Kramaric", "Bruno Petkovic", "Marko Livaja",
              "Mislav Orsic", "Ante Budimir"],
    },
    "Brazil": {
        "G": ["Alisson", "Ederson", "Weverton"],
        "D": ["Dani Alves", "Danilo", "Alex Sandro", "Alex Telles", "Bremer",
              "Eder Militao", "Marquinhos", "Thiago Silva"],
        "M": ["Casemiro", "Fred", "Fabinho", "Bruno Guimaraes", "Lucas Paqueta", "Everton Ribeiro"],
        "F": ["Neymar", "Vinicius Junior", "Richarlison", "Raphinha", "Antony",
              "Rodrygo", "Gabriel Jesus", "Gabriel Martinelli", "Pedro"],
    },
    "Serbia": {
        "G": ["Vanja Milinkovic-Savic", "Predrag Rajkovic", "Marko Dmitrovic"],
        "D": ["Nikola Milenkovic", "Strahinja Pavlovic", "Milos Veljkovic", "Stefan Mitrovic",
              "Filip Mladenovic", "Strahinja Erakovic", "Srdjan Babic"],
        "M": ["Sergej Milinkovic-Savic", "Filip Kostic", "Nemanja Gudelj", "Sasa Lukic",
              "Nemanja Maksimovic", "Marko Grujic", "Ivan Ilic", "Andrija Zivkovic", "Darko Lazovic"],
        "F": ["Aleksandar Mitrovic", "Dusan Vlahovic", "Dusan Tadic", "Luka Jovic", "Nemanja Radonjic"],
    },
    "Switzerland": {
        "G": ["Yann Sommer", "Gregor Kobel", "Jonas Omlin"],
        "D": ["Manuel Akanji", "Nico Elvedi", "Fabian Schar", "Ricardo Rodriguez",
              "Silvan Widmer", "Eray Comert", "Edimilson Fernandes"],
        "M": ["Granit Xhaka", "Remo Freuler", "Denis Zakaria", "Fabian Frei",
              "Djibril Sow", "Michel Aebischer", "Ardon Jashari"],
        "F": ["Xherdan Shaqiri", "Breel Embolo", "Haris Seferovic", "Ruben Vargas",
              "Noah Okafor", "Renato Steffen", "Fabian Rieder"],
    },
    "Cameroon": {
        "G": ["Andre Onana", "Devis Epassy", "Simon Ngapandouetnbu"],
        "D": ["Jean-Charles Castelletto", "Nicolas Nkoulou", "Collins Fai", "Nouhou Tolo",
              "Olivier Mbaizo", "Enzo Ebosse", "Christopher Wooh"],
        "M": ["Andre-Frank Zambo Anguissa", "Pierre Kunde", "Martin Hongla", "Gael Ondoua",
              "Samuel Oum Gouet", "Olivier Ntcham"],
        "F": ["Vincent Aboubakar", "Eric Maxim Choupo-Moting", "Karl Toko Ekambi", "Bryan Mbeumo",
              "Georges-Kevin Nkoudou", "Jean-Pierre Nsame", "Christian Bassogog", "Nicolas Moumi Ngamaleu"],
    },
    "Portugal": {
        "G": ["Diogo Costa", "Rui Patricio", "Jose Sa"],
        "D": ["Pepe", "Ruben Dias", "Danilo Pereira", "Antonio Silva", "Diogo Dalot",
              "Nuno Mendes", "Joao Cancelo", "Raphael Guerreiro"],
        "M": ["Bruno Fernandes", "Bernardo Silva", "Joao Palhinha", "Ruben Neves",
              "William Carvalho", "Vitinha", "Matheus Nunes", "Otavio"],
        "F": ["Cristiano Ronaldo", "Joao Felix", "Rafael Leao", "Goncalo Ramos",
              "Andre Silva", "Ricardo Horta"],
    },
    "Ghana": {
        "G": ["Lawrence Ati-Zigi", "Manaf Nurudeen", "Danlad Ibrahim"],
        "D": ["Daniel Amartey", "Alexander Djiku", "Mohammed Salisu", "Gideon Mensah",
              "Tariq Lamptey", "Denis Odoi", "Baba Rahman", "Alidu Seidu"],
        "M": ["Thomas Partey", "Mohammed Kudus", "Salis Abdul Samed", "Elisha Owusu",
              "Daniel-Kofi Kyereh", "Majeed Ashimeru", "Andre Ayew"],
        "F": ["Inaki Williams", "Jordan Ayew", "Antoine Semenyo", "Kamaldeen Sulemana",
              "Osman Bukari", "Abdul Fatawu Issahaku"],
    },
    "Uruguay": {
        "G": ["Sergio Rochet", "Fernando Muslera", "Sebastian Sosa"],
        "D": ["Diego Godin", "Jose Maria Gimenez", "Ronald Araujo", "Sebastian Coates",
              "Martin Caceres", "Mathias Olivera", "Guillermo Varela"],
        "M": ["Federico Valverde", "Rodrigo Bentancur", "Matias Vecino", "Lucas Torreira",
              "Mauro Arambarri", "Manuel Ugarte", "Nicolas de la Cruz", "Facundo Pellistri",
              "Giorgian de Arrascaeta"],
        "F": ["Luis Suarez", "Edinson Cavani", "Darwin Nunez", "Maxi Gomez", "Agustin Canobbio"],
    },
    "South Korea": {
        "G": ["Kim Seung-gyu", "Jo Hyeon-woo", "Song Bum-keun"],
        "D": ["Kim Min-jae", "Kim Young-gwon", "Kwon Kyung-won", "Kim Moon-hwan",
              "Kim Tae-hwan", "Hong Chul", "Cho Yu-min", "Kim Jin-su"],
        "M": ["Lee Jae-sung", "Jung Woo-young", "Hwang In-beom", "Son Jun-ho",
              "Paik Seung-ho", "Lee Kang-in", "Kwon Chang-hoon", "Na Sang-ho"],
        "F": ["Son Heung-min", "Hwang Hee-chan", "Hwang Ui-jo", "Cho Gue-sung", "Song Min-kyu"],
    },
}

_GRP = {"G": "GK", "D": "DEF", "M": "MID", "F": "ATT"}

# Players lost to injury (pre-tournament names are already out of the squads above;
# the in-tournament ones make this a hindsight-only, opt-in adjustment).
INJURED_2022 = {
    "Brazil": {"Philippe Coutinho", "Gabriel Jesus", "Alex Telles"},
    "Portugal": {"Diogo Jota", "Pedro Neto", "Ricardo Pereira", "Nuno Mendes"},
}


def _pool(edition: int = EDITION) -> pd.DataFrame:
    p = load_fifa_players()
    p = p[p["edition"] == edition].copy()
    p["_n"] = p["short_name"].map(_norm)
    return p


def _match_squad(team: str, squad: dict, pool: pd.DataFrame, exclude: set | None = None):
    exclude = exclude or set()
    cand = pool[pool["nationality"] == team]
    names = cand["_n"].tolist()
    ovr = cand["overall"].astype(float).tolist()
    surn = [n.split()[-1] if n else "" for n in names]
    matched, miss = [], []
    for key, code in _GRP.items():
        for name in squad.get(key, []):
            if _norm(name) in exclude:
                continue
            pn = _ALIAS.get(_norm(name), _norm(name))
            sn = pn.split()[-1]
            same = [i for i, s in enumerate(surn) if s == sn]
            if same:
                i = max(same, key=lambda i: fuzz.token_set_ratio(pn, names[i]))
                matched.append((ovr[i], code))
                continue
            hit = process.extractOne(pn, names, scorer=fuzz.token_set_ratio)
            if hit and hit[1] >= FUZZY_CUTOFF:
                matched.append((ovr[hit[2]], code))
            else:
                miss.append(name)
    return matched, miss


def confirmed_table_2022(edition: int = EDITION, drop_injured: bool = False):
    """Strength table for the 32 teams from confirmed squads, proxy fallback when undermatched."""
    pool = _pool(edition)
    proxy = edition_table_filled(edition)
    rows, report = {}, []
    for team, squad in SQUADS_2022.items():
        exclude = {_norm(x) for x in INJURED_2022.get(team, ())} if drop_injured else None
        matched, miss = _match_squad(team, squad, pool, exclude)
        n_named = sum(len(v) for v in squad.values())
        if len(matched) >= MIN_MATCHED:
            s = _strength_from_players(matched)
            pr = proxy.loc[team] if team in proxy.index else None
            for col in _STRENGTH:
                if pd.isna(s.get(col)) and pr is not None:
                    s[col] = float(pr[col])
            rows[team] = s
            report.append((team, len(matched), n_named, "confirmed"))
        else:
            src = proxy.loc[team] if team in proxy.index else None
            rows[team] = {c: float(src[c]) for c in _STRENGTH} if src is not None else None
            report.append((team, len(matched), n_named, "proxy"))
    tbl = pd.DataFrame.from_dict({k: v for k, v in rows.items() if v}, orient="index")[_STRENGTH]
    tbl.index.name = "team"
    rep = pd.DataFrame(report, columns=["team", "matched", "named", "source"])
    return tbl, rep


def main():
    tbl, rep = confirmed_table_2022()
    n_conf = (rep["source"] == "confirmed").sum()
    print(f"confirmed-squad table: {n_conf}/{len(rep)} teams from real squads, "
          f"{len(rep) - n_conf} via proxy fallback")
    print(rep.sort_values("matched").to_string(index=False))


if __name__ == "__main__":
    main()
