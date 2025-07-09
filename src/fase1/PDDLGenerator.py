import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import re
import json
import yaml
import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Union
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
    branching_factor: Tuple[int, int]
    depth_constraints: Tuple[int, int]


class ActionType(Enum):
    """Tipi di azioni possibili nella quest"""
    MOVE = "move"
    TAKE = "take"
    USE = "use"
    TALK = "talk"
    FIGHT = "fight"
    SOLVE = "solve"
    EXPLORE = "explore"


def _parse_from_dict(data: dict) -> LoreDocument:
    """Crea LoreDocument da dizionario"""
    return LoreDocument(
        quest_description=data.get('quest_description', ''),
        initial_state=data.get('initial_state', ''),
        goal_state=data.get('goal_state', ''),
        world_context=data.get('world_context', ''),
        characters=data.get('characters', []),
        locations=data.get('locations', []),
        items=data.get('items', []),
        obstacles=data.get('obstacles', []),
        branching_factor=tuple(data.get('branching_factor', [2, 4])),
        depth_constraints=tuple(data.get('depth_constraints', [3, 10]))
    )


def _parse_from_text(lore_text: str) -> LoreDocument:
    """Parser testuale per formato legacy"""
    lines = lore_text.strip().split('\n')

    quest_description = ""
    initial_state = ""
    goal_state = ""
    world_context = ""
    characters = []
    locations = []
    items = []
    obstacles = []
    branching_factor = (2, 4)
    depth_constraints = (3, 10)

    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Identifica sezioni
        if line.upper().startswith("QUEST:"):
            current_section = "quest"
            quest_description = line[6:].strip()
        elif line.upper().startswith("INITIAL:"):
            current_section = "initial"
            initial_state = line[8:].strip()
        elif line.upper().startswith("GOAL:"):
            current_section = "goal"
            goal_state = line[5:].strip()
        elif line.upper().startswith("WORLD:"):
            current_section = "world"
            world_context = line[6:].strip()
        elif line.upper().startswith("CHARACTERS:"):
            current_section = "characters"
            characters = [c.strip() for c in line[11:].split(',') if c.strip()]
        elif line.upper().startswith("LOCATIONS:"):
            current_section = "locations"
            locations = [l.strip() for l in line[10:].split(',') if l.strip()]
        elif line.upper().startswith("ITEMS:"):
            current_section = "items"
            items = [i.strip() for i in line[6:].split(',') if i.strip()]
        elif line.upper().startswith("OBSTACLES:"):
            current_section = "obstacles"
            obstacles = [o.strip() for o in line[10:].split(',') if o.strip()]
        elif line.upper().startswith("BRANCHING:"):
            values = line[10:].strip().split('-')
            if len(values) == 2:
                branching_factor = (int(values[0]), int(values[1]))
        elif line.upper().startswith("DEPTH:"):
            values = line[6:].strip().split('-')
            if len(values) == 2:
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


class PDDLGenerator:
    """
    Generatore di file PDDL da documenti Lore
    Versione integrata con tutti i miglioramenti
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
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
        self.metadata = {}

    def generate_complete_pddl(self, lore_input: Union[str, dict]) -> Tuple[str, str, Dict[str, any]]:
        """
        Genera sia il domain che il problem PDDL da un input Lore
        Versione migliorata con validazione e metadata

        Args:
            lore_input: Testo, dizionario o path al file Lore

        Returns:
            - domain_pddl: String del domain file
            - problem_pddl: String del problem file
            - metadata: Dictionary con statistiche e validazione
        """
        try:
            # 1. Parse del documento Lore (supporta multi-formato)
            self.logger.info("Parsing Lore document...")
            lore = self.parse_lore_document(lore_input)

            # 2. Genera predicati (versione migliorata)
            self.logger.info("Generating predicates...")
            self.predicates = self.generate_predicates(lore)

            # 3. Genera azioni
            self.logger.info("Generating actions...")
            self.actions = self.generate_actions(lore)

            # Import qui per evitare circolarità
            try:
                from src.fase1.ActionGenerator import AdvancedActionGenerator
                action_gen = AdvancedActionGenerator(lore)
                advanced_actions = action_gen.generate_smart_actions()
                self.actions.extend(advanced_actions)
            except ImportError:
                self.logger.warning("AdvancedActionGenerator not found, using basic actions only")

            # 4. Genera stati iniziali e goal
            self.logger.info("Generating initial and goal states...")
            self.initial_state = self.generate_initial_state(lore)
            self.goal_state = self.generate_goal_state(lore)

            # 5. Costruisci i file PDDL
            domain_pddl = self.generate_domain_pddl(lore)
            problem_pddl = self.generate_problem_pddl(lore)

            # 6. Valida il risultato (se disponibile)
            is_valid = True
            errors = []
            warnings = []

            try:
                from src.fase1.Validator import PDDLConstraintsValidator
                validator = PDDLConstraintsValidator(domain_pddl, problem_pddl, lore)
                is_valid, errors, warnings = validator.validate_all()
            except ImportError:
                self.logger.warning("PDDLConstraintsValidator not found, skipping validation")

            # 7. Prepara metadata
            self.metadata = {
                'is_valid': is_valid,
                'validation_errors': errors,
                'validation_warnings': warnings,
                'statistics': {
                    'num_predicates': len(self.predicates),
                    'num_actions': len(self.actions),
                    'num_objects': sum(len(objs) for objs in self.objects.values()),
                    'initial_state_size': len(self.initial_state),
                    'goal_size': len(self.goal_state),
                    'estimated_complexity': self._estimate_complexity(lore)
                },
                'lore_summary': {
                    'quest': lore.quest_description,
                    'characters': len(lore.characters),
                    'locations': len(lore.locations),
                    'branching': f"{lore.branching_factor[0]}-{lore.branching_factor[1]}",
                    'depth': f"{lore.depth_constraints[0]}-{lore.depth_constraints[1]}"
                }
            }

            return domain_pddl, problem_pddl, self.metadata

        except Exception as e:
            self.logger.error(f"Error generating PDDL: {str(e)}")
            raise

    def parse_lore_document(self, lore_input: Union[str, dict]) -> LoreDocument:
        """
        Parsa il documento Lore - supporta testo, JSON e YAML
        """
        # Se è già un dizionario
        if isinstance(lore_input, dict):
            return _parse_from_dict(lore_input)

        # Se è una stringa
        lore_text = lore_input.strip()

        # Prova JSON
        if lore_text.startswith('{'):
            try:
                data = json.loads(lore_text)
                return _parse_from_dict(data)
            except json.JSONDecodeError:
                pass

        # Prova YAML
        if ':' in lore_text and not lore_text.upper().startswith('QUEST:'):
            try:
                data = yaml.safe_load(lore_text)
                if isinstance(data, dict):
                    return _parse_from_dict(data)
            except yaml.YAMLError:
                pass

        # Altrimenti usa il parser testuale
        return _parse_from_text(lore_text)

    def generate_predicates(self, lore: LoreDocument) -> Set[str]:
        """
        Genera i predicati PDDL basati sul contenuto del Lore
        Versione migliorata con predicati avanzati
        """
        predicates = set()

        # Predicati di base per avventure
        predicates.add("(at ?obj ?loc)")
        predicates.add("(player-at ?loc)")
        predicates.add("(has ?item)")
        predicates.add("(connected ?loc1 ?loc2)")
        predicates.add("(alive ?char)")
        predicates.add("(talked-to ?char)")
        predicates.add("(solved ?obstacle)")
        predicates.add("(explored ?loc)")
        predicates.add("(locked ?loc)")
        predicates.add("(unlocked ?loc)")

        # Predicati avanzati per narrativa complessa
        if len(lore.characters) > 1:
            predicates.add("(trusts ?char1 ?char2)")
            predicates.add("(allied-with ?char)")
            predicates.add("(hostile-to ?char)")

        # Predicati per branching narrativo
        predicates.add("(choice-made ?choice)")
        predicates.add("(quest-path ?path)")
        predicates.add("(karma-level ?level)")

        # Predicati specifici per items
        for item in lore.items:
            normalized = self.normalize_name(item)
            predicates.add(f"(has-{normalized})")
            predicates.add(f"(used-{normalized})")

            # Predicati speciali per tipi di oggetti
            if 'key' in item.lower():
                predicates.add(f"(opens-with-{normalized} ?loc)")

        # Predicati specifici per ostacoli
        for obstacle in lore.obstacles:
            normalized = self.normalize_name(obstacle)
            predicates.add(f"(blocked-by-{normalized} ?loc)")
            predicates.add(f"(vulnerable-to-{normalized} ?item)")

        return predicates

    def generate_actions(self, lore: LoreDocument) -> List[str]:
        """
        Genera le azioni PDDL base
        Le azioni avanzate vengono aggiunte da AdvancedActionGenerator
        """
        actions = []

        # Azione MOVE migliorata
        move_action = """(:action move
    :parameters (?from ?to)
    :precondition (and 
        (player-at ?from)
        (connected ?from ?to)
        (not (locked ?to))
    )
    :effect (and 
        (not (player-at ?from))
        (player-at ?to)
        (explored ?to)
    )
)"""
        actions.append(move_action)

        # Azione TAKE migliorata
        take_action = """(:action take
    :parameters (?item ?loc)
    :precondition (and 
        (player-at ?loc)
        (at ?item ?loc)
        (not (has ?item))
    )
    :effect (and 
        (not (at ?item ?loc))
        (has ?item)
    )
)"""
        actions.append(take_action)

        # Azione TALK con effetti
        talk_action = """(:action talk
    :parameters (?char ?loc)
    :precondition (and 
        (player-at ?loc)
        (at ?char ?loc)
        (alive ?char)
        (not (talked-to ?char))
    )
    :effect (and 
        (talked-to ?char)
        (choice-made talk-to-?char)
    )
)"""
        actions.append(talk_action)

        # Azioni USE per ogni item
        for item in lore.items:
            item_name = self.normalize_name(item)

            # Azione use generica
            use_action = f"""(:action use-{item_name}
    :parameters (?target ?loc)
    :precondition (and 
        (player-at ?loc)
        (has {item_name})
    )
    :effect (and 
        (used-{item_name})
    )
)"""
            actions.append(use_action)

            # Azioni specifiche per tipo di item
            if 'key' in item.lower():
                unlock_action = f"""(:action unlock-with-{item_name}
    :parameters (?loc)
    :precondition (and 
        (player-at ?loc)
        (has {item_name})
        (locked ?loc)
        (opens-with-{item_name} ?loc)
    )
    :effect (and 
        (not (locked ?loc))
        (unlocked ?loc)
        (used-{item_name})
    )
)"""
                actions.append(unlock_action)

        # Azioni per risolvere ostacoli
        for obstacle in lore.obstacles:
            obstacle_name = self.normalize_name(obstacle)

            # Azione solve generica
            solve_action = f"""(:action solve-{obstacle_name}
    :parameters (?loc)
    :precondition (and 
        (player-at ?loc)
        (blocked-by-{obstacle_name} ?loc)
    )
    :effect (and 
        (not (blocked-by-{obstacle_name} ?loc))
        (solved {obstacle_name})
        (choice-made solved-{obstacle_name})
    )
)"""
            actions.append(solve_action)

            # Azione combat se l'ostacolo è un nemico
            if any(enemy_word in obstacle.lower() for enemy_word in ['dragon', 'enemy', 'guard', 'monster']):
                fight_action = f"""(:action fight-{obstacle_name}
    :parameters (?loc ?weapon)
    :precondition (and 
        (player-at ?loc)
        (blocked-by-{obstacle_name} ?loc)
        (has ?weapon)
        (vulnerable-to-{obstacle_name} ?weapon)
    )
    :effect (and 
        (not (blocked-by-{obstacle_name} ?loc))
        (not (alive {obstacle_name}))
        (solved {obstacle_name})
        (karma-level decreased)
    )
)"""
                actions.append(fight_action)

        return actions

    def generate_initial_state(self, lore: LoreDocument) -> List[str]:
        """
        Genera lo stato iniziale PDDL
        Versione migliorata con posizionamento intelligente
        """
        initial = []

        # Reset oggetti per il problema
        self.objects = {
            'characters': set(lore.characters),
            'locations': set(lore.locations),
            'items': set(lore.items),
            'obstacles': set(lore.obstacles)
        }

        # Posizione iniziale del player
        if lore.locations:
            start_loc = self.normalize_name(lore.locations[0])
            initial.append(f"(player-at {start_loc})")
            initial.append(f"(explored {start_loc})")

        # Connessioni tra location (grafo migliorato)
        for i in range(len(lore.locations) - 1):
            loc1 = self.normalize_name(lore.locations[i])
            loc2 = self.normalize_name(lore.locations[i + 1])
            initial.append(f"(connected {loc1} {loc2})")
            initial.append(f"(connected {loc2} {loc1})")

        # Aggiungi connessioni extra per branching
        if len(lore.locations) > 4:
            # Scorciatoia
            initial.append(
                f"(connected {self.normalize_name(lore.locations[1])} {self.normalize_name(lore.locations[-2])})")

        # Posiziona oggetti in modo narrativamente sensato
        item_placement = self._smart_item_placement(lore)
        for item, location in item_placement.items():
            initial.append(f"(at {self.normalize_name(item)} {self.normalize_name(location)})")

        # Posiziona personaggi
        for i, char in enumerate(lore.characters):
            # Distribuisci in location diverse
            loc_idx = (i + 1) % len(lore.locations) if len(lore.locations) > 1 else 0
            loc = self.normalize_name(lore.locations[loc_idx])
            initial.append(f"(at {self.normalize_name(char)} {loc})")
            initial.append(f"(alive {self.normalize_name(char)})")

            # Stati iniziali dei personaggi
            if 'enemy' in char.lower() or 'dragon' in char.lower():
                initial.append(f"(hostile-to {self.normalize_name(char)})")

        # Ostacoli attivi
        for i, obstacle in enumerate(lore.obstacles):
            if lore.locations:
                # Posiziona ostacoli nelle location più avanzate
                loc_idx = min(i + len(lore.locations) // 2, len(lore.locations) - 1)
                loc = self.normalize_name(lore.locations[loc_idx])

                if 'door' in obstacle.lower() or 'gate' in obstacle.lower():
                    initial.append(f"(locked {loc})")
                else:
                    initial.append(f"(blocked-by-{self.normalize_name(obstacle)} {loc})")

                # Vulnerabilità degli ostacoli
                if 'dragon' in obstacle.lower():
                    for item in lore.items:
                        if 'sword' in item.lower() or 'weapon' in item.lower():
                            initial.append(
                                f"(vulnerable-to-{self.normalize_name(obstacle)} {self.normalize_name(item)})")

        # Stati narrativi iniziali
        initial.append("(karma-level neutral)")
        initial.append("(quest-path main)")

        return initial

    def generate_goal_state(self, lore: LoreDocument) -> List[str]:
        """
        Genera lo stato goal PDDL
        Versione migliorata basata sull'analisi del goal narrativo
        """
        goals = []

        # Analizza il goal narrativo
        goal_keywords = self._extract_goal_keywords(lore.goal_state)

        # Goal di posizione (se menzionato)
        if lore.locations:
            # Se il goal menziona "tornare", l'obiettivo è la prima location
            if 'return' in goal_keywords or 'back' in goal_keywords:
                goals.append(f"(player-at {self.normalize_name(lore.locations[0])})")
            # Altrimenti, l'obiettivo è l'ultima location
            else:
                final_loc = self.normalize_name(lore.locations[-1])
                goals.append(f"(player-at {final_loc})")

        # Goal basati su keywords
        if 'defeat' in goal_keywords or 'kill' in goal_keywords:
            for char in lore.characters:
                if any(enemy in char.lower() for enemy in ['dragon', 'enemy', 'villain', 'boss']):
                    goals.append(f"(not (alive {self.normalize_name(char)}))")

        if 'save' in goal_keywords or 'rescue' in goal_keywords:
            for char in lore.characters:
                if any(friend in char.lower() for friend in ['princess', 'prince', 'prisoner']):
                    goals.append(f"(talked-to {self.normalize_name(char)})")

        if 'find' in goal_keywords or 'collect' in goal_keywords:
            for item in lore.items:
                if any(valuable in item.lower() for valuable in ['treasure', 'artifact', 'crystal', 'sword']):
                    goals.append(f"(has {self.normalize_name(item)})")

        # Goal di completamento ostacoli
        for obstacle in lore.obstacles:
            goals.append(f"(solved {self.normalize_name(obstacle)})")

        # Goal di esplorazione se la quest è esplorativa
        if 'explore' in goal_keywords or 'discover' in goal_keywords:
            for loc in lore.locations:
                goals.append(f"(explored {self.normalize_name(loc)})")

        # Assicura che ci siano abbastanza goal per la complessità richiesta
        min_goals = max(3, lore.depth_constraints[0] // 3)
        if len(goals) < min_goals:
            # Aggiungi goal di interazione
            for char in lore.characters[:2]:
                if f"(talked-to {self.normalize_name(char)})" not in goals:
                    goals.append(f"(talked-to {self.normalize_name(char)})")

        return goals

    def generate_domain_pddl(self, lore: LoreDocument) -> str:
        """Genera il file domain PDDL completo"""
        domain_name = "quest-domain"

        # Header con documentazione
        domain_pddl = f";; Domain PDDL per: {lore.quest_description}\n"
        domain_pddl += f";; Generato da QuestMaster PDDLGenerator\n"
        domain_pddl += f";; Branching factor: {lore.branching_factor[0]}-{lore.branching_factor[1]}\n"
        domain_pddl += f";; Depth constraints: {lore.depth_constraints[0]}-{lore.depth_constraints[1]}\n\n"
        domain_pddl += f"(define (domain {domain_name})\n\n"

        # Requirements
        domain_pddl += "    ;; PDDL Requirements\n"
        domain_pddl += "    (:requirements :strips :typing :negative-preconditions)\n\n"

        # Types
        domain_pddl += "    ;; Type definitions\n"
        domain_pddl += "    (:types\n"
        domain_pddl += "        location\n"
        domain_pddl += "        character\n"
        domain_pddl += "        item\n"
        domain_pddl += "        obstacle\n"
        domain_pddl += "    )\n\n"

        # Predicates
        domain_pddl += "    ;; Predicates\n"
        domain_pddl += "    (:predicates\n"
        for pred in sorted(self.predicates):
            domain_pddl += f"        {pred}    ;; {self._get_predicate_comment(pred)}\n"
        domain_pddl += "    )\n\n"

        # Actions
        domain_pddl += "    ;; Actions\n"
        for action in self.actions:
            domain_pddl += f"    {action}\n\n"

        domain_pddl += ")\n"

        return domain_pddl

    def generate_problem_pddl(self, lore: LoreDocument) -> str:
        """Genera il file problem PDDL completo"""
        problem_name = "quest-problem"
        domain_name = "quest-domain"

        # Header
        problem_pddl = f";; Problem PDDL per: {lore.quest_description}\n"
        problem_pddl += f";; Stato iniziale: {lore.initial_state}\n"
        problem_pddl += f";; Goal: {lore.goal_state}\n\n"
        problem_pddl += f"(define (problem {problem_name})\n"
        problem_pddl += f"    (:domain {domain_name})\n\n"

        # Objects
        problem_pddl += "    ;; Objects\n"
        problem_pddl += "    (:objects\n"

        # Locations
        if self.objects['locations']:
            problem_pddl += "        ;; Locations\n"
            for loc in sorted(self.objects['locations']):
                problem_pddl += f"        {self.normalize_name(loc)} - location\n"

        # Characters
        if self.objects['characters']:
            problem_pddl += "        ;; Characters\n"
            for char in sorted(self.objects['characters']):
                problem_pddl += f"        {self.normalize_name(char)} - character\n"

        # Items
        if self.objects['items']:
            problem_pddl += "        ;; Items\n"
            for item in sorted(self.objects['items']):
                problem_pddl += f"        {self.normalize_name(item)} - item\n"

        # Obstacles
        if self.objects['obstacles']:
            problem_pddl += "        ;; Obstacles\n"
            for obstacle in sorted(self.objects['obstacles']):
                problem_pddl += f"        {self.normalize_name(obstacle)} - obstacle\n"

        problem_pddl += "    )\n\n"

        # Initial state
        problem_pddl += "    ;; Initial state\n"
        problem_pddl += "    (:init\n"
        for state in sorted(self.initial_state):
            problem_pddl += f"        {state}\n"
        problem_pddl += "    )\n\n"

        # Goal state
        problem_pddl += "    ;; Goal\n"
        problem_pddl += "    (:goal\n        (and\n"
        for goal in sorted(self.goal_state):
            problem_pddl += f"            {goal}\n"
        problem_pddl += "        )\n    )\n"

        problem_pddl += ")\n"

        return problem_pddl

    # Metodi helper privati
    def _get_predicate_comment(self, predicate: str) -> str:
        """Genera commenti esplicativi per i predicati"""
        comments = {
            "at": "Posizione di oggetti/personaggi",
            "player-at": "Posizione del giocatore",
            "has": "Oggetti nell'inventario",
            "connected": "Connessioni tra location",
            "alive": "Stato vitale",
            "talked-to": "Dialoghi completati",
            "solved": "Ostacoli risolti",
            "explored": "Location visitate",
            "locked": "Location bloccate",
            "trusts": "Relazioni di fiducia",
            "allied-with": "Alleanze",
            "hostile-to": "Ostilità",
            "choice-made": "Scelte narrative",
            "karma-level": "Livello morale"
        }

        for key, comment in comments.items():
            if key in predicate:
                return comment
        return "Predicato specifico della quest"

    def _extract_goal_keywords(self, goal_text: str) -> Set[str]:
        """Estrae parole chiave dal goal per generazione intelligente"""
        keywords = set()
        important_words = [
            'defeat', 'kill', 'destroy',
            'save', 'rescue', 'protect',
            'find', 'collect', 'gather',
            'explore', 'discover', 'uncover',
            'return', 'escape', 'back'
        ]

        goal_lower = goal_text.lower()
        for word in important_words:
            if word in goal_lower:
                keywords.add(word)

        return keywords

    def _smart_item_placement(self, lore: LoreDocument) -> Dict[str, str]:
        """Posiziona items in modo narrativamente sensato"""
        placement = {}

        if not lore.locations:
            return placement

        for i, item in enumerate(lore.items):
            item_lower = item.lower()

            # Logica di posizionamento basata sul tipo
            if 'key' in item_lower or 'map' in item_lower:
                # Oggetti iniziali vicino all'inizio
                placement[item] = lore.locations[min(1, len(lore.locations) - 1)]
            elif 'weapon' in item_lower or 'sword' in item_lower:
                # Armi a metà percorso
                mid = len(lore.locations) // 2
                placement[item] = lore.locations[mid]
            elif 'treasure' in item_lower or 'artifact' in item_lower:
                # Tesori verso la fine
                placement[item] = lore.locations[-2 if len(lore.locations) > 1 else -1]
            else:
                # Distribuzione uniforme per altri oggetti
                loc_idx = i % len(lore.locations)
                placement[item] = lore.locations[loc_idx]

        return placement

    def _estimate_complexity(self, lore: LoreDocument) -> float:
        """Stima la complessità della quest (1-10)"""
        factors = {
            'locations': len(lore.locations) * 1.0,
            'characters': len(lore.characters) * 1.5,
            'items': len(lore.items) * 1.2,
            'obstacles': len(lore.obstacles) * 2.0,
            'branching': (lore.branching_factor[0] + lore.branching_factor[1]) / 2,
            'depth': (lore.depth_constraints[0] + lore.depth_constraints[1]) / 2
        }

        complexity = sum(factors.values()) / 10.0
        return min(10.0, max(1.0, complexity))

    def normalize_name(self, name: str) -> str:
        """Normalizza i nomi per PDDL"""
        return re.sub(r'[^a-z0-9\-]', '', name.lower().replace(' ', '-'))


# Esempio di utilizzo standalone
if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Esempio di Lore
    lore_example = """
QUEST: Il giocatore deve recuperare la spada magica per sconfiggere il drago
INITIAL: Il giocatore si trova nel villaggio con solo una mappa
GOAL: Il giocatore deve arrivare alla torre del drago con la spada magica e sconfiggere il drago
WORLD: Un regno fantasy con villaggi, foreste e torri misteriose
CHARACTERS: Mago Saggio, Mercante, Cavaliere Nero, Drago Antico
LOCATIONS: Villaggio, Foresta, Ponte, Torre del Mago, Tana del Drago
ITEMS: Spada Magica, Mappa, Pozione, Chiave della Torre, Scudo
OBSTACLES: Drago Antico, Porta Magica, Cavaliere Nero
BRANCHING: 2-4
DEPTH: 5-10
"""

    # Genera PDDL
    generator = PDDLGenerator()
    domain, problem, metadata = generator.generate_complete_pddl(lore_example)

    print("=== DOMAIN PDDL ===")
    print(domain)
    print("\n=== PROBLEM PDDL ===")
    print(problem)
    print("\n=== METADATA ===")
    print(json.dumps(metadata, indent=2))

    # Salva i file
    with open("quest_domain.pddl", "w") as f:
        f.write(domain)

    with open("quest_problem.pddl", "w") as f:
        f.write(problem)

    print("\nFile salvati: quest_domain.pddl, quest_problem.pddl")