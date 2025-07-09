import itertools
from typing import List, Dict

from src.fase1.PDDLGenerator import LoreDocument


class AdvancedActionGenerator:
    """Generatore avanzato di azioni PDDL con logica complessa"""

    def __init__(self, lore: LoreDocument):
        self.lore = lore
        self.action_templates = self._initialize_action_templates()

    def _initialize_action_templates(self) -> Dict[str, str]:
        """Template per azioni complesse"""
        return {
            'move_conditional': """(:action move-when-{condition}
    :parameters (?from - location ?to - location)
    :precondition (and 
        (player-at ?from)
        (connected ?from ?to)
        {additional_preconditions}
    )
    :effect (and 
        (not (player-at ?from))
        (player-at ?to)
        (explored ?to)
        {additional_effects}
    )
)""",

            'use_item_on_target': """(:action use-{item}-on-{target}
    :parameters (?loc - location)
    :precondition (and 
        (player-at ?loc)
        (has {item})
        {target_conditions}
    )
    :effect (and 
        (not (has {item}))
        (used-{item})
        {target_effects}
    )
)""",

            'complex_interaction': """(:action {action_name}
    :parameters ({parameters})
    :precondition (and 
        {preconditions}
    )
    :effect (and 
        {effects}
    )
)"""
        }

    def generate_smart_actions(self) -> List[str]:
        """Genera azioni intelligenti basate sul contesto narrativo"""
        actions = []

        # Azioni di movimento contestuali
        actions.extend(self._generate_contextual_movement())

        # Azioni di uso oggetti combinate
        actions.extend(self._generate_item_combinations())

        # Azioni narrative complesse
        actions.extend(self._generate_narrative_actions())

        # Azioni con branching narrativo
        actions.extend(self._generate_branching_actions())

        return actions

    def _generate_contextual_movement(self) -> List[str]:
        """Genera movimenti che richiedono condizioni specifiche"""
        actions = []

        # Movimento che richiede oggetti specifici
        for item in self.lore.items:
            if 'key' in item.lower() or 'pass' in item.lower():
                action = self.action_templates['move_conditional'].format(
                    condition=f"with-{self.normalize_name(item)}",
                    additional_preconditions=f"(has {self.normalize_name(item)})",
                    additional_effects="(achievement unlocked-area)"
                )
                actions.append(action)

        # Movimento che richiede di aver parlato con personaggi
        for char in self.lore.characters:
            if 'guard' in char.lower() or 'guide' in char.lower():
                action = self.action_templates['move_conditional'].format(
                    condition=f"after-{self.normalize_name(char)}",
                    additional_preconditions=f"(talked-to {self.normalize_name(char)})",
                    additional_effects=f"(guided-by {self.normalize_name(char)})"
                )
                actions.append(action)

        return actions

    def _generate_item_combinations(self) -> List[str]:
        """Genera azioni per combinazioni di oggetti"""
        actions = []

        # Combinazioni di 2 oggetti
        for item1, item2 in itertools.combinations(self.lore.items, 2):
            if self._items_can_combine(item1, item2):
                action = f"""(:action combine-{self.normalize_name(item1)}-with-{self.normalize_name(item2)}
    :parameters (?loc - location)
    :precondition (and 
        (player-at ?loc)
        (has {self.normalize_name(item1)})
        (has {self.normalize_name(item2)})
    )
    :effect (and 
        (not (has {self.normalize_name(item1)}))
        (not (has {self.normalize_name(item2)}))
        (has combined-{self.normalize_name(item1)}-{self.normalize_name(item2)})
    )
)"""
                actions.append(action)

        return actions

    def _generate_narrative_actions(self) -> List[str]:
        """Genera azioni che avanzano la narrativa"""
        actions = []

        # Azioni di dialogo con conseguenze
        for char in self.lore.characters:
            # Dialogo che sblocca informazioni
            action = f"""(:action learn-from-{self.normalize_name(char)}
    :parameters (?loc - location)
    :precondition (and 
        (player-at ?loc)
        (at {self.normalize_name(char)} ?loc)
        (alive {self.normalize_name(char)})
        (not (knows-secret-{self.normalize_name(char)}))
    )
    :effect (and 
        (talked-to {self.normalize_name(char)})
        (knows-secret-{self.normalize_name(char)})
        (quest-progress increased)
    )
)"""
            actions.append(action)

        # Azioni di esplorazione con scoperte
        for loc in self.lore.locations:
            if 'hidden' in loc.lower() or 'secret' in loc.lower():
                action = f"""(:action discover-secret-in-{self.normalize_name(loc)}
    :parameters ()
    :precondition (and 
        (player-at {self.normalize_name(loc)})
        (explored {self.normalize_name(loc)})
        (not (discovered-secret-{self.normalize_name(loc)}))
    )
    :effect (and 
        (discovered-secret-{self.normalize_name(loc)})
        (revealed new-path)
    )
)"""
                actions.append(action)

        return actions

    def _generate_branching_actions(self) -> List[str]:
        """Genera azioni che creano branch narrativi"""
        actions = []

        # Scelte morali
        moral_choices = [
            ("help", "harm"),
            ("save", "abandon"),
            ("trust", "betray")
        ]

        for positive, negative in moral_choices:
            for char in self.lore.characters[:2]:  # Limita a primi 2 personaggi
                # Scelta positiva
                action_pos = f"""(:action {positive}-{self.normalize_name(char)}
    :parameters (?loc - location)
    :precondition (and 
        (player-at ?loc)
        (at {self.normalize_name(char)} ?loc)
        (alive {self.normalize_name(char)})
        (not (made-choice-about-{self.normalize_name(char)}))
    )
    :effect (and 
        (made-choice-about-{self.normalize_name(char)})
        ({positive}ed-{self.normalize_name(char)})
        (karma-increased)
        (relationship-{self.normalize_name(char)} positive)
    )
)"""

                # Scelta negativa
                action_neg = f"""(:action {negative}-{self.normalize_name(char)}
    :parameters (?loc - location)
    :precondition (and 
        (player-at ?loc)
        (at {self.normalize_name(char)} ?loc)
        (alive {self.normalize_name(char)})
        (not (made-choice-about-{self.normalize_name(char)}))
    )
    :effect (and 
        (made-choice-about-{self.normalize_name(char)})
        ({negative}ed-{self.normalize_name(char)})
        (karma-decreased)
        (relationship-{self.normalize_name(char)} negative)
    )
)"""
                actions.extend([action_pos, action_neg])

        return actions

    def _items_can_combine(self, item1: str, item2: str) -> bool:
        """Logica per determinare se due oggetti possono combinarsi"""
        combinable_pairs = [
            ('sword', 'stone'),
            ('potion', 'herb'),
            ('key', 'lock'),
            ('map', 'compass'),
            ('fire', 'ice'),
            ('light', 'dark')
        ]

        item1_lower = item1.lower()
        item2_lower = item2.lower()

        for pair in combinable_pairs:
            if (pair[0] in item1_lower and pair[1] in item2_lower) or \
                    (pair[1] in item1_lower and pair[0] in item2_lower):
                return True

        return False

    def normalize_name(self, name: str) -> str:
        """Normalizza i nomi per PDDL"""
        import re
        return re.sub(r'[^a-z0-9\-]', '', name.lower().replace(' ', '-'))

    def validate_action_complexity(self, actions: List[str]) -> Dict[str, int]:
        """Valida che le azioni rispettino i branching factor"""
        stats = {
            'total_actions': len(actions),
            'average_preconditions': 0,
            'average_effects': 0,
            'branching_score': 0
        }

        for action in actions:
            precond_count = action.count('and')
            effect_count = action.count('effect')
            stats['average_preconditions'] += precond_count
            stats['average_effects'] += effect_count

        if actions:
            stats['average_preconditions'] //= len(actions)
            stats['average_effects'] //= len(actions)

        # Calcola branching score basato sui constraints
        min_branch, max_branch = self.lore.branching_factor
        if min_branch <= len(actions) <= max_branch * 3:
            stats['branching_score'] = 100
        else:
            stats['branching_score'] = max(0, 100 - abs(len(actions) - max_branch * 2) * 10)

        return stats