import re
import json
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum


@dataclass
class LoreDocument:
    """Struttura dati per il documento Lore"""
    quest_description: str
    initial_state: str
    goal_state: str
    world_context: str
    characters: List[str]
    locations: List[str]
    items: List[str]
    obstacles: List[str]
    branching_factor: Tuple[int, int]  # (min, max)
    depth_constraints: Tuple[int, int]  # (min, max)


class ActionType(Enum):
    """Tipi di azioni possibili nella quest"""
    MOVE = "move"
    TAKE = "take"
    USE = "use"
    TALK = "talk"
    FIGHT = "fight"
    SOLVE = "solve"
    EXPLORE = "explore"


class PDDLGenerator:
    """Generatore di file PDDL da documenti Lore"""

    def __init__(self):
        self.predicates = set()
        self.actions = []
        self.objects = {
            'characters': set(),
            'locations': set(),
            'items': set(),
            'obstacles': set()
        }
        self.initial_state = []
        self.goal_state = []

    def parse_lore_document(self, lore_text: str) -> LoreDocument:
        """
        Parsa il documento Lore e estrae le informazioni strutturate
        """
        # Parsing semplificato - in un'implementazione reale useresti NLP più sofisticato
        lines = lore_text.strip().split('\n')

        quest_description = ""
        initial_state = ""
        goal_state = ""
        world_context = ""
        characters = []
        locations = []
        items = []
        obstacles = []
        branching_factor = (2, 4)  # default
        depth_constraints = (3, 10)  # default

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Identifica sezioni
            if line.startswith("QUEST:"):
                current_section = "quest"
                quest_description = line[6:].strip()
            elif line.startswith("INITIAL:"):
                current_section = "initial"
                initial_state = line[8:].strip()
            elif line.startswith("GOAL:"):
                current_section = "goal"
                goal_state = line[5:].strip()
            elif line.startswith("WORLD:"):
                current_section = "world"
                world_context = line[6:].strip()
            elif line.startswith("CHARACTERS:"):
                current_section = "characters"
                characters = [c.strip() for c in line[11:].split(',')]
            elif line.startswith("LOCATIONS:"):
                current_section = "locations"
                locations = [l.strip() for l in line[10:].split(',')]
            elif line.startswith("ITEMS:"):
                current_section = "items"
                items = [i.strip() for i in line[6:].split(',')]
            elif line.startswith("OBSTACLES:"):
                current_section = "obstacles"
                obstacles = [o.strip() for o in line[10:].split(',')]
            elif line.startswith("BRANCHING:"):
                values = line[10:].strip().split('-')
                branching_factor = (int(values[0]), int(values[1]))
            elif line.startswith("DEPTH:"):
                values = line[6:].strip().split('-')
                depth_constraints = (int(values[0]), int(values[1]))
            else:
                # Continua la sezione corrente
                if current_section == "quest":
                    quest_description += " " + line
                elif current_section == "initial":
                    initial_state += " " + line
                elif current_section == "goal":
                    goal_state += " " + line
                elif current_section == "world":
                    world_context += " " + line

        return LoreDocument(
            quest_description=quest_description,
            initial_state=initial_state,
            goal_state=goal_state,
            world_context=world_context,
            characters=characters,
            locations=locations,
            items=items,
            obstacles=obstacles,
            branching_factor=branching_factor,
            depth_constraints=depth_constraints
        )

    def generate_predicates(self, lore: LoreDocument) -> Set[str]:
        """
        Genera i predicati PDDL basati sul contenuto del Lore
        """
        predicates = set()

        # Predicati di base per avventure
        predicates.add("(at ?obj ?loc)")  # Un oggetto è in una location
        predicates.add("(player-at ?loc)")  # Il player è in una location
        predicates.add("(has ?item)")  # Il player ha un item
        predicates.add("(connected ?loc1 ?loc2)")  # Due location sono connesse
        predicates.add("(alive ?char)")  # Un personaggio è vivo
        predicates.add("(talked-to ?char)")  # Hai parlato con un personaggio
        predicates.add("(solved ?obstacle)")  # Un ostacolo è risolto
        predicates.add("(explored ?loc)")  # Una location è stata esplorata

        # Predicati specifici per items
        for item in lore.items:
            predicates.add(f"(has-{self.normalize_name(item)})")
            predicates.add(f"(used-{self.normalize_name(item)})")

        # Predicati specifici per ostacoli
        for obstacle in lore.obstacles:
            predicates.add(f"(blocked-by-{self.normalize_name(item)})")

        return predicates

    def generate_actions(self, lore: LoreDocument) -> List[str]:
        """
        Genera le azioni PDDL basate sul contenuto del Lore
        """
        actions = []

        # Azione MOVE - spostarsi tra location
        move_action = """(:action move
    :parameters (?from ?to)
    :precondition (and 
        (player-at ?from)
        (connected ?from ?to)
        (not (blocked-by-obstacle ?to))
    )
    :effect (and 
        (not (player-at ?from))
        (player-at ?to)
        (explored ?to)
    )
)"""
        actions.append(move_action)

        # Azione TAKE - prendere oggetti
        take_action = """(:action take
    :parameters (?item ?loc)
    :precondition (and 
        (player-at ?loc)
        (at ?item ?loc)
    )
    :effect (and 
        (not (at ?item ?loc))
        (has ?item)
    )
)"""
        actions.append(take_action)

        # Azione TALK - parlare con personaggi
        talk_action = """(:action talk
    :parameters (?char ?loc)
    :precondition (and 
        (player-at ?loc)
        (at ?char ?loc)
        (alive ?char)
    )
    :effect (talked-to ?char)
)"""
        actions.append(talk_action)

        # Azioni specifiche per items
        for item in lore.items:
            item_name = self.normalize_name(item)
            use_action = f"""(:action use-{item_name}
    :parameters (?target)
    :precondition (has {item_name})
    :effect (used-{item_name})
)"""
            actions.append(use_action)

        # Azioni per risolvere ostacoli
        for obstacle in lore.obstacles:
            obstacle_name = self.normalize_name(obstacle)
            solve_action = f"""(:action solve-{obstacle_name}
    :parameters (?loc)
    :precondition (and 
        (player-at ?loc)
        (blocked-by-{obstacle_name} ?loc)
    )
    :effect (and 
        (not (blocked-by-{obstacle_name} ?loc))
        (solved {obstacle_name})
    )
)"""
            actions.append(solve_action)

        return actions

    def generate_initial_state(self, lore: LoreDocument) -> List[str]:
        """
        Genera lo stato iniziale PDDL
        """
        initial = []

        # Posizione iniziale del player (prima location)
        if lore.locations:
            initial.append(f"(player-at {self.normalize_name(lore.locations[0])})")

        # Posiziona oggetti nelle location
        for i, item in enumerate(lore.items):
            if i < len(lore.locations):
                loc = lore.locations[i % len(lore.locations)]
                initial.append(f"(at {self.normalize_name(item)} {self.normalize_name(loc)})")

        # Posiziona personaggi
        for i, char in enumerate(lore.characters):
            if i < len(lore.locations):
                loc = lore.locations[i % len(lore.locations)]
                initial.append(f"(at {self.normalize_name(char)} {self.normalize_name(loc)})")
                initial.append(f"(alive {self.normalize_name(char)})")

        # Connessioni tra location (grafo lineare semplice)
        for i in range(len(lore.locations) - 1):
            loc1 = lore.locations[i].lower().replace(' ', '-')
            loc2 = lore.locations[i + 1].lower().replace(' ', '-')
            initial.append(f"(connected {self.normalize_name(loc1)} {self.normalize_name(loc2)})")
            initial.append(f"(connected {self.normalize_name(loc1)} {self.normalize_name(loc2)})")


        # Ostacoli attivi
        for obstacle in lore.obstacles:
            if lore.locations:
                # Posiziona ostacolo in una location casuale
                loc = lore.locations[-1].lower().replace(' ', '-')  # Ultima location
                initial.append(f"(blocked-by-{self.normalize_name(obstacle)} {self.normalize_name(loc)})")

        return initial

    def generate_goal_state(self, lore: LoreDocument) -> List[str]:
        """
        Genera lo stato goal PDDL
        """
        goals = []

        # Goal di base: essere nell'ultima location
        if lore.locations:
            final_loc = lore.locations[-1].lower().replace(' ', '-')
            goals.append(f"(player-at {self.normalize_name(final_loc)})")

        # Goal: avere parlato con tutti i personaggi
        for char in lore.characters:
            goals.append(f"(talked-to {self.normalize_name(char)})")

        # Goal: aver risolto tutti gli ostacoli
        for obstacle in lore.obstacles:
            goals.append(f"(solved {self.normalize_name(obstacle)})")

        # Goal: aver esplorato tutte le location
        for loc in lore.locations:
            goals.append(f"(explored {self.normalize_name(loc)})")

        return goals

    def generate_domain_pddl(self, lore: LoreDocument) -> str:
        """
        Genera il file domain PDDL completo
        """
        domain_name = "quest-domain"

        # Header
        domain_pddl = f";; Domain PDDL per la quest: {lore.quest_description}\n"
        domain_pddl += f";; Generato automaticamente dal sistema QuestMaster\n\n"
        domain_pddl += f"(define (domain {domain_name})\n\n"

        # Requirements
        domain_pddl += "    ;; Specifica le funzionalità PDDL necessarie\n"
        domain_pddl += "    (:requirements :strips :typing)\n\n"

        # Types
        domain_pddl += "    ;; Definizione dei tipi di oggetti nel dominio\n"
        domain_pddl += "    (:types\n"
        domain_pddl += "        location     ;; Luoghi della quest\n"
        domain_pddl += "        character    ;; Personaggi della storia\n"
        domain_pddl += "        item         ;; Oggetti che si possono raccogliere\n"
        domain_pddl += "        obstacle     ;; Ostacoli da superare\n"
        domain_pddl += "    )\n\n"

        # Predicates
        domain_pddl += "    ;; Predicati che descrivono le proprietà del mondo\n"
        domain_pddl += "    (:predicates\n"
        predicates = self.generate_predicates(lore)
        for pred in sorted(predicates):
            domain_pddl += f"        {pred}    ;; {self._get_predicate_comment(pred)}\n"
        domain_pddl += "    )\n\n"

        # Actions
        domain_pddl += "    ;; Azioni disponibili nella quest\n"
        actions = self.generate_actions(lore)
        for action in actions:
            domain_pddl += f"    {action}\n\n"

        domain_pddl += ")\n"

        return domain_pddl

    def generate_problem_pddl(self, lore: LoreDocument) -> str:
        """
        Genera il file problem PDDL completo
        """
        problem_name = "quest-problem"
        domain_name = "quest-domain"

        # Header
        problem_pddl = f";; Problem PDDL per la quest: {lore.quest_description}\n"
        problem_pddl += f";; Stato iniziale e goal della quest\n\n"
        problem_pddl += f"(define (problem {problem_name})\n"
        problem_pddl += f"    (:domain {domain_name})\n\n"

        # Objects
        problem_pddl += "    ;; Oggetti presenti nella quest\n"
        problem_pddl += "    (:objects\n"

        # Locations
        problem_pddl += "        ;; Luoghi della quest\n"
        for loc in lore.locations:
            problem_pddl += f"        {self.normalize_name(loc)} - location\n"

        # Characters
        problem_pddl += "        ;; Personaggi della storia\n"
        for char in lore.characters:
            problem_pddl += f"        {self.normalize_name(char)} - character\n"

        # Items
        problem_pddl += "        ;; Oggetti raccoglibili\n"
        for item in lore.items:
            problem_pddl += f"        {self.normalize_name(item)} - item\n"

        # Obstacles
        problem_pddl += "        ;; Ostacoli da superare\n"
        for obstacle in lore.obstacles:
            problem_pddl += f"        {self.normalize_name(obstacle)} - obstacle\n"

        problem_pddl += "    )\n\n"

        # Initial state
        problem_pddl += "    ;; Stato iniziale della quest\n"
        problem_pddl += "    (:init\n"
        initial_state = self.generate_initial_state(lore)
        for state in initial_state:
            problem_pddl += f"        {state}    ;; {self._get_state_comment(state)}\n"
        problem_pddl += "    )\n\n"

        # Goal state
        problem_pddl += "    ;; Obiettivo della quest\n"
        problem_pddl += "    (:goal\n        (and\n"
        goal_state = self.generate_goal_state(lore)
        for goal in goal_state:
            problem_pddl += f"            {goal}    ;; {self._get_goal_comment(goal)}\n"
        problem_pddl += "        )\n    )\n"

        problem_pddl += ")\n"

        return problem_pddl

    def _get_predicate_comment(self, predicate: str) -> str:
        """Genera commenti esplicativi per i predicati"""
        if "at" in predicate:
            return "Posizione di oggetti e personaggi"
        elif "player-at" in predicate:
            return "Posizione corrente del giocatore"
        elif "has" in predicate:
            return "Oggetti posseduti dal giocatore"
        elif "connected" in predicate:
            return "Connessioni tra location"
        elif "alive" in predicate:
            return "Stato di vita dei personaggi"
        elif "talked-to" in predicate:
            return "Conversazioni completate"
        elif "solved" in predicate:
            return "Ostacoli risolti"
        elif "explored" in predicate:
            return "Location esplorate"
        else:
            return "Predicato custom per la quest"

    def _get_state_comment(self, state: str) -> str:
        """Genera commenti per gli stati iniziali"""
        if "player-at" in state:
            return "Posizione di partenza del giocatore"
        elif "at" in state:
            return "Posizione iniziale di oggetti/personaggi"
        elif "connected" in state:
            return "Connessione tra location"
        elif "alive" in state:
            return "Personaggio inizialmente vivo"
        elif "blocked-by" in state:
            return "Ostacolo attivo"
        else:
            return "Stato iniziale custom"

    def _get_goal_comment(self, goal: str) -> str:
        """Genera commenti per i goal"""
        if "player-at" in goal:
            return "Destinazione finale"
        elif "talked-to" in goal:
            return "Conversazione richiesta"
        elif "solved" in goal:
            return "Ostacolo da risolvere"
        elif "explored" in goal:
            return "Location da esplorare"
        else:
            return "Obiettivo custom"

    def generate_complete_pddl(self, lore_text: str) -> Tuple[str, str]:
        """
        Genera sia il domain che il problem PDDL da un testo Lore
        """
        lore = self.parse_lore_document(lore_text)
        domain_pddl = self.generate_domain_pddl(lore)
        problem_pddl = self.generate_problem_pddl(lore)

        return domain_pddl, problem_pddl

    def normalize_name(self, name: str) -> str:
        return re.sub(r'[^a-z0-9\-]', '', name.lower().replace(' ', '-'))


# Esempio di utilizzo
if __name__ == "__main__":
    # Esempio di documento Lore
    lore_example = """
QUEST: Il giocatore deve recuperare la spada magica per sconfiggere il drago
INITIAL: Il giocatore si trova nel villaggio con solo una mappa
GOAL: Il giocatore deve arrivare alla torre del drago con la spada magica e sconfiggere il drago
WORLD: Un regno fantasy con villaggi, foreste e torri misteriose
CHARACTERS: Mago, Mercante, Drago
LOCATIONS: Villaggio, Foresta, Torre
ITEMS: Spada Magica, Mappa, Pozione
OBSTACLES: Drago, Porta Magica
"""

    # Genera PDDL
    generator = PDDLGenerator()
    domain, problem = generator.generate_complete_pddl(lore_example)

    print("=== DOMAIN PDDL ===")
    print(domain)
    print("\n=== PROBLEM PDDL ===")
    print(problem)

    # Salva i file
    with open("quest_domain.pddl", "w") as f:
        f.write(domain)

    with open("quest_problem.pddl", "w") as f:
        f.write(problem)