"""Plan sections and the builders that turn them into event terms.

A builder takes the section's value from the plan and returns {term name: EventTerm}. Sections build in
registration order (built-ins first), so a section registered later runs after the built-in ones at each
reset. Register a new section, or replace a built-in one, from a module the plan lists under "extras":

    from robolab.core.sim2real import register

    @register("lid_spring")
    def build_lid_spring(cfg):
        return {"dr_lid_spring": EventTerm(func=my_func, mode="startup", params=cfg)}
"""
SECTIONS = {}


def register(section, replace=False):
    def deco(builder):
        if section in SECTIONS and not replace and SECTIONS[section] is not builder:
            raise KeyError(f"plan section {section!r} is already registered; pass replace=True to override it")
        SECTIONS[section] = builder
        return builder

    return deco
