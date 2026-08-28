import asyncio
import json
import os
import typer
from datetime import datetime, timezone, UTC
from uuid import UUID
from typing import Optional, List
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich import print as rprint
from sqlalchemy.sql.expression import func
from sqlmodel import select

from blindmind.db import init_db, get_async_session, save_concept, create_run, search_concepts, get_stats, delete_concept, get_projects
from blindmind.models import Concept, RunStatus, EvolutionRun, Lineage
from blindmind.engine import EvolutionEngine
from blindmind.config import settings
from blindmind.logging import logger
from blindmind.llm import llm_engine

app = typer.Typer(
    help="BlindMind: Elite Evolutionary Concept Refinement using LLMs.",
    rich_markup_mode="rich"
)
console = Console()

_active_project = "default"

# --- Visual Helpers ---

def score_bar(value: float, max_val: float = 10, width: int = 10) -> str:
    filled = int((value / max_val) * width)
    empty = width - filled
    if value >= 7:
        color = "green"
    elif value >= 5:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{'━' * filled}[/{color}][dim]{'╌' * empty}[/dim] {value:.1f}"

def mini_bar(value: int, max_val: int = 10) -> str:
    filled = value
    empty = max_val - value
    if value >= 7:
        color = "green"
    elif value >= 5:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"

def welcome_banner():
    dna = "[dim cyan]╔══╗[/dim cyan]"
    art = (
        "[bold cyan]  ┌──────────────────────────────────┐[/bold cyan]\n"
        "[bold cyan]  │[/bold cyan]  [bold white]B L I N D M I N D[/bold white]   [dim]E L I T E[/dim]  [bold cyan]│[/bold cyan]\n"
        "[bold cyan]  │[/bold cyan]  [dim]Evolutionary Concept Refinement[/dim]  [bold cyan]│[/bold cyan]\n"
        "[bold cyan]  └──────────────────────────────────┘[/bold cyan]"
    )
    console.print(art)

async def ensure_setup() -> str:
    """Setup API keys, init DB, and return the active project name."""
    global _active_project

    found_keys = []
    if settings.openai_api_key: found_keys.append("OpenAI")
    if settings.anthropic_api_key: found_keys.append("Anthropic")
    if settings.gemini_api_key: found_keys.append("Gemini")
    if settings.openrouter_api_key: found_keys.append("OpenRouter")

    if not found_keys:
        rprint("\n[bold yellow]No API keys found.[/bold yellow]")
        key = Prompt.ask("Enter [cyan]OPENAI_API_KEY[/cyan] (or blank to try others)")
        if key:
            settings.openai_api_key = key
        else:
            anthropic_key = Prompt.ask("Enter [cyan]ANTHROPIC_API_KEY[/cyan]")
            if anthropic_key: settings.anthropic_api_key = anthropic_key

        if settings.openai_api_key or settings.anthropic_api_key:
            settings.save_local_env()
            rprint("[green]Key saved.[/green]")
            found_keys.append("Configured")
    else:
        rprint(f"  [dim]API:[/dim] {', '.join(found_keys)}")

    if found_keys:
        from blindmind.llm import llm_engine
        llm_engine.providers = await llm_engine._get_available_providers()
        llm_engine._initialized = True

    await init_db()

    # Check for existing projects
    async for session in get_async_session():
        projects = await get_projects(session)

        if projects:
            # Projects exist: ask which to load
            rprint(f"\n  [dim]Projects:[/dim]")
            for i, p in enumerate(projects, 1):
                s = await get_stats(session, project=p)
                rprint(f"    [bold]{i}[/bold]. {p} [dim]({s['total']} concepts, {s['domains']} domains, gen {s['generations']})[/dim]")
            rprint(f"    [bold]{len(projects)+1}[/bold]. [dim]Create new project[/dim]")

            pick = Prompt.ask(f"\n  [bold]Load project[/bold] [dim](1-{len(projects)+1})[/dim]", default="1")
            try:
                pick_num = int(pick)
                if 1 <= pick_num <= len(projects):
                    _active_project = projects[pick_num - 1]
                elif pick_num == len(projects) + 1:
                    new_name = Prompt.ask("  Project name")
                    if new_name.strip():
                        _active_project = new_name.strip()
                    rprint(f"  [green]Created: {_active_project}[/green]")
                else:
                    _active_project = projects[0]
            except ValueError:
                # Treat input as project name
                if pick.strip() in projects:
                    _active_project = pick.strip()
                else:
                    _active_project = projects[0]
        else:
            # No projects: offer to load seeds
            if Confirm.ask("[yellow]Load default seed concepts?[/yellow]", default=True):
                seeds = [
                    {"domain": "Biology", "title": "Mycelial Networks", "description": "Decentralized information exchange in fungi.", "tags": "biology,networks,decentralized"},
                    {"domain": "Physics", "title": "Quantum Entanglement", "description": "Instant state correlation across distance.", "tags": "physics,quantum,correlation"},
                    {"domain": "Finance", "title": "DeFi Liquidity Pools", "description": "Automated market making via smart contracts.", "tags": "finance,defi,automation"},
                    {"domain": "Philosophy", "title": "Stoicism", "description": "Focusing on what is within one's control.", "tags": "philosophy,mindset,resilience"},
                ]
                for s in seeds:
                    await save_concept(session, Concept(**s, generation=0, project="default"))
                rprint("[green]Seeds loaded.[/green]")
            _active_project = "default"

    return _active_project

# --- Core Logic ---

async def run_evolution_logic(generations: int, population: int, threshold: float = None, temperature: float = None, model: str = None, project: str = "default"):
    if threshold is not None:
        settings.critic_threshold = threshold
    if temperature is not None:
        settings.variation_temperature = temperature
    if model is not None:
        settings.litellm_model = model
        llm_engine.set_model_override(model)

    async for session in get_async_session():
        run_obj = await create_run(session, config_json=settings.model_dump_json(), total_generations=generations, population_size=population, project=project)
        current_directive = run_obj.latest_directive

        rprint(Panel(
            f"[dim]Project:[/dim] [bold]{project}[/bold]  [dim]Threshold:[/dim] {settings.critic_threshold}  [dim]Temp:[/dim] {settings.variation_temperature}  [dim]Model:[/dim] {settings.litellm_model}",
            title="[bold cyan]Evolution Run[/bold cyan]",
            border_style="cyan",
        ))

        try:
            for gen in range(1, generations + 1):
                run_obj.current_generation = gen
                run_obj.updated_at = datetime.now(UTC)
                session.add(run_obj)
                await session.commit()

                engine = EvolutionEngine(session, directive=current_directive, project=project)

                with Progress(
                    SpinnerColumn(spinner_name="dots"),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task(f"[cyan]Gen {gen}/{generations}[/cyan] Evolving {population} candidates...", total=None)
                    survivors = await engine.run_generation_cycle(gen, population)

                if not survivors:
                    rprint(f"[red]Gen {gen}: No candidates passed (threshold: {engine.adaptive_threshold:.1f}). Try lowering it.[/red]")
                    break

                rprint(f"\n[bold]{'─' * 50}[/bold]")
                rprint(f"[bold yellow]Generation {gen}[/bold yellow] [dim]({len(survivors)} elite candidates)[/dim]")
                if current_directive:
                    rprint(f"[dim]Directive: {current_directive[:80]}[/dim]")
                rprint(f"[bold]{'─' * 50}[/bold]")

                weighted_directives = []
                accepted_count = 0
                for i, (mutation, critique, parent_ids, m_type) in enumerate(survivors, 1):
                    # Build visual score display
                    scores_display = (
                        f"  Novelty    {mini_bar(critique.conceptual_novelty)} {critique.conceptual_novelty}\n"
                        f"  Feasible   {mini_bar(critique.feasibility)} {critique.feasibility}\n"
                        f"  Utility    {mini_bar(critique.utility)} {critique.utility}\n"
                        f"  Sem. Jump  {mini_bar(critique.semantic_jump)} {critique.semantic_jump}\n"
                        f"  Prior Art  {mini_bar(10 - critique.prior_art_overlap)} {critique.prior_art_overlap} [dim]({'Novel' if critique.prior_art_overlap <= 3 else 'Known' if critique.prior_art_overlap >= 7 else 'Mixed'})[/dim]\n"
                        f"\n"
                        f"  [bold]Composite   {score_bar(critique.composite_score)}[/bold]"
                    )

                    impl_section = ""
                    if critique.implementation_path:
                        impl_section = f"\n\n[dim]How to build:[/dim] {critique.implementation_path}"

                    flaws_section = ""
                    if critique.fatal_flaws:
                        flaws_section = "\n[red]Flaws:[/red] " + "; ".join(critique.fatal_flaws)

                    type_icon = {"CROSSOVER": "x", "POINT_MUTATION": "~", "INVERSION": "!", "WILDCARD": "*"}.get(m_type, "?")
                    type_label = {"CROSSOVER": "Crossover", "POINT_MUTATION": "Mutation", "INVERSION": "Inversion", "WILDCARD": "Wildcard"}.get(m_type, m_type)

                    console.print(Panel(
                        f"[bold white]{mutation.title}[/bold white]\n"
                        f"[magenta]{mutation.domain}[/magenta]\n\n"
                        f"{mutation.description}\n\n"
                        f"{scores_display}"
                        f"{flaws_section}{impl_section}\n\n"
                        f"[dim italic]{critique.evolutionary_directive}[/dim italic]",
                        title=f"[bold][{type_icon}] {type_label}[/bold] [dim]({i}/{len(survivors)})[/dim]",
                        border_style="blue" if critique.composite_score >= 7 else "yellow" if critique.composite_score >= 5 else "red",
                        padding=(1, 2),
                    ))

                    score_input = Prompt.ask(
                        "[bold]Score[/bold] [dim](1-10 keep, Enter skip)[/dim]",
                        default="0"
                    )
                    try:
                        user_score = int(score_input)
                    except ValueError:
                        user_score = 0

                    if 1 <= user_score <= 10:
                        blended = (critique.composite_score + user_score) / 2.0
                        tags_input = Prompt.ask("Tags [dim](optional)[/dim]", default="")
                        new_concept = Concept(
                            project=project,
                            domain=mutation.domain,
                            title=mutation.title,
                            description=mutation.description,
                            generation=gen,
                            fitness_score=blended,
                            tags=tags_input if tags_input else None,
                        )
                        await save_concept(session, new_concept, parent_ids=parent_ids, mutation_type=m_type)
                        accepted_count += 1
                        weighted_directives.append((critique.evolutionary_directive, critique.composite_score))
                        rprint(f"  [green]Retained[/green] [dim](fitness: {blended:.1f})[/dim]")
                    else:
                        rprint("  [dim]Skipped[/dim]")

                    run_obj.concepts_generated = run_obj.concepts_generated + 1

                if weighted_directives:
                    current_directive = EvolutionEngine.synthesize_directives(weighted_directives)
                    run_obj.latest_directive = current_directive

                run_obj.concepts_retained = run_obj.concepts_retained + accepted_count
                rprint(f"\n  [bold green]Gen {gen} complete:[/bold green] {accepted_count} retained")

            run_obj.status = RunStatus.COMPLETED
        except KeyboardInterrupt:
            rprint("\n[yellow]Interrupted. Progress saved.[/yellow]")
            run_obj.status = RunStatus.CANCELLED
        except Exception as e:
            logger.exception("Run failed")
            run_obj.status = RunStatus.FAILED
            rprint(f"\n[bold red]Error:[/bold red] {e}")
        finally:
            run_obj.updated_at = datetime.now(UTC)
            session.add(run_obj)
            await session.commit()

            from blindmind.llm import llm_engine
            s = llm_engine.stats.summary
            summary_parts = [f"{s['total_calls']} LLM calls"]
            if s['failed']:
                summary_parts.append(f"[red]{s['failed']} failed[/red]")
            if s['input_tokens'] + s['output_tokens']:
                summary_parts.append(f"{s['input_tokens']+s['output_tokens']:,} tokens")
            if s['avg_latency_ms']:
                summary_parts.append(f"avg {s['avg_latency_ms']}ms")
            if hasattr(engine, 'adaptive_threshold') and engine.adaptive_threshold != settings.critic_threshold:
                summary_parts.append(f"threshold {settings.critic_threshold:.1f} -> {engine.adaptive_threshold:.1f}")

            rprint(f"\n  [dim]{' | '.join(summary_parts)}[/dim]")

async def list_concepts_logic(generation: Optional[int], limit: int, domain: str = None, min_fitness: float = None, project: str = None):
    async for session in get_async_session():
        if domain or min_fitness or project:
            results = await search_concepts(session, domain=domain, min_fitness=min_fitness, generation=generation, limit=limit, project=project)
        else:
            statement = select(Concept)
            if generation is not None:
                statement = statement.where(Concept.generation == generation)
            statement = statement.order_by(Concept.created_at.desc()).limit(limit)
            results = (await session.execute(statement)).scalars().all()

        if not results:
            rprint("[dim]No concepts found.[/dim]")
            return

        table = Table(border_style="dim", show_edge=False, pad_edge=False)
        table.add_column("ID", style="dim", width=8)
        table.add_column("G", justify="right", style="cyan", width=2)
        table.add_column("Domain", style="magenta", max_width=20)
        table.add_column("Title", style="bold", ratio=2)
        table.add_column("Fit", justify="right", width=5)
        table.add_column("Tags", style="dim", max_width=20, overflow="ellipsis")

        for c in results:
            fit = f"{c.fitness_score:.1f}" if c.fitness_score else "[dim]-[/dim]"
            fit_style = "green" if c.fitness_score and c.fitness_score >= 7 else "yellow" if c.fitness_score and c.fitness_score >= 5 else ""
            table.add_row(
                c.short_id, str(c.generation), c.domain, c.title,
                f"[{fit_style}]{fit}[/{fit_style}]" if fit_style else fit,
                c.tags or ""
            )
        console.print(table)
        rprint(f"  [dim]{len(results)} concepts[/dim]")

async def search_concepts_logic(query: str, domain: str = None, min_fitness: float = None, limit: int = 20, project: str = None):
    async for session in get_async_session():
        results = await search_concepts(session, query=query, domain=domain, min_fitness=min_fitness, limit=limit, project=project)

        if not results:
            rprint(f"[dim]No results for '{query}'[/dim]")
            return

        table = Table(title=f"[dim]Search:[/dim] {query}", border_style="dim", show_edge=False)
        table.add_column("ID", style="dim", width=8)
        table.add_column("G", justify="right", style="cyan", width=2)
        table.add_column("Domain", style="magenta", max_width=20)
        table.add_column("Title", style="bold", ratio=2)
        table.add_column("Fit", justify="right", width=5)

        for c in results:
            fit = f"{c.fitness_score:.1f}" if c.fitness_score else "-"
            table.add_row(c.short_id, str(c.generation), c.domain, c.title, fit)
        console.print(table)

async def stats_logic(project: str = None):
    async for session in get_async_session():
        s = await get_stats(session, project=project)

        if s["total"] == 0:
            rprint("[dim]Empty latent space.[/dim]")
            return

        # Compact stats in a panel
        has_scores = s['max_fitness'] > 0
        fitness_line = (
            f"Fitness: [green]{s['max_fitness']}[/green] best  [dim]|[/dim]  "
            f"[yellow]{s['avg_fitness']}[/yellow] avg  [dim]|[/dim]  "
            f"[red]{s['min_fitness']}[/red] worst"
        ) if has_scores else "Fitness: [dim]no scored concepts yet[/dim]"

        stats_text = (
            f"[bold]{s['total']}[/bold] concepts  [dim]|[/dim]  "
            f"[cyan]{s['seeds']}[/cyan] seeds  [dim]|[/dim]  "
            f"[green]{s['evolved']}[/green] evolved  [dim]|[/dim]  "
            f"[magenta]{s['domains']}[/magenta] domains  [dim]|[/dim]  "
            f"Gen [yellow]{s['generations']}[/yellow]\n\n"
            f"{fitness_line}"
        )
        console.print(Panel(stats_text, title=f"[bold]Latent Space[/bold]{f' ({project})' if project else ''}", border_style="cyan"))

        if "gen_distribution" in s and s["gen_distribution"]:
            max_count = max(s["gen_distribution"].values())
            rprint()
            for gen, count in sorted(s["gen_distribution"].items()):
                bar_len = int((count / max(max_count, 1)) * 25)
                bar = "[cyan]" + "█" * bar_len + "[/cyan]"
                label = "seed" if gen == 0 else f"gen {gen}"
                rprint(f"  {label:>6}  {bar} {count}")

async def pick_concept_id(project: str, prompt_label: str = "Pick") -> Optional[str]:
    """Show concept list and let user pick one by row number or short ID. Returns short ID or None."""
    async for session in get_async_session():
        results = (await session.execute(
            select(Concept).where(Concept.project == project).order_by(Concept.created_at.desc()).limit(30)
        )).scalars().all()

        if not results:
            rprint("  [dim]No concepts in this project.[/dim]")
            return None

        table = Table(border_style="dim", show_edge=False, pad_edge=False)
        table.add_column("#", style="dim", width=3)
        table.add_column("ID", style="dim", width=8)
        table.add_column("G", justify="right", style="cyan", width=2)
        table.add_column("Domain", style="magenta", max_width=15)
        table.add_column("Title", style="bold", ratio=2)
        table.add_column("Fit", justify="right", width=5)

        for i, c in enumerate(results, 1):
            fit = f"{c.fitness_score:.1f}" if c.fitness_score else "[dim]-[/dim]"
            table.add_row(str(i), c.short_id, str(c.generation), c.domain, c.title, fit)
        console.print(table)

        pick = Prompt.ask(f"  [bold]{prompt_label}[/bold] [dim](# or ID, empty to cancel)[/dim]", default="")
        if not pick.strip():
            return None

        try:
            row_num = int(pick.strip())
            if 1 <= row_num <= len(results):
                return results[row_num - 1].short_id
        except ValueError:
            pass

        return pick.strip()


async def view_concept_logic(concept_id: str):
    async for session in get_async_session():
        all_concepts = (await session.execute(select(Concept))).scalars().all()
        concept = next((c for c in all_concepts if str(c.id).startswith(concept_id)), None)
        if not concept:
            rprint(f"[red]Not found: '{concept_id}'[/red]")
            return

        fit_display = score_bar(concept.fitness_score) if concept.fitness_score else "[dim]unscored[/dim]"

        # Gather lineage context
        lineage_text = ""
        parent_links = (await session.execute(
            select(Lineage).where(Lineage.child_id == concept.id)
        )).scalars().all()
        child_links = (await session.execute(
            select(Lineage).where(Lineage.parent_id == concept.id)
        )).scalars().all()

        if parent_links:
            parent_lines = []
            for l in parent_links:
                p = next((c for c in all_concepts if c.id == l.parent_id), None)
                if p:
                    icon = {"CROSSOVER": "x", "POINT_MUTATION": "~", "INVERSION": "!", "WILDCARD": "*"}.get(l.mutation_type, "?")
                    parent_lines.append(f"  [{icon}] [cyan]{p.title}[/cyan] [dim]({p.domain}, Gen {p.generation})[/dim]")
            if parent_lines:
                lineage_text += "\n[dim]Parents:[/dim]\n" + "\n".join(parent_lines)

        if child_links:
            child_lines = []
            for l in child_links:
                ch = next((c for c in all_concepts if c.id == l.child_id), None)
                if ch:
                    icon = {"CROSSOVER": "x", "POINT_MUTATION": "~", "INVERSION": "!", "WILDCARD": "*"}.get(l.mutation_type, "?")
                    fit_ch = f" {ch.fitness_score:.1f}" if ch.fitness_score else ""
                    child_lines.append(f"  [{icon}] [green]{ch.title}[/green] [dim]({ch.domain}, Gen {ch.generation}{fit_ch})[/dim]")
            if child_lines:
                lineage_text += "\n[dim]Children:[/dim]\n" + "\n".join(child_lines)

        console.print(Panel(
            f"[bold white]{concept.title}[/bold white]\n"
            f"[magenta]{concept.domain}[/magenta] [dim]|[/dim] Gen {concept.generation} [dim]|[/dim] {concept.project}\n\n"
            f"{concept.description}\n\n"
            f"Fitness: {fit_display}\n"
            f"[dim]Tags: {concept.tags or 'none'}[/dim]"
            f"{lineage_text}\n\n"
            f"[dim]ID: {concept.id}[/dim]\n"
            f"[dim]{concept.created_at.strftime('%Y-%m-%d %H:%M')}[/dim]",
            border_style="blue",
        ))

async def delete_concept_logic(concept_id: str):
    async for session in get_async_session():
        all_concepts = (await session.execute(select(Concept))).scalars().all()
        concept = next((c for c in all_concepts if str(c.id).startswith(concept_id)), None)
        if not concept:
            rprint(f"[red]Not found: '{concept_id}'[/red]")
            return
        rprint(f"  [yellow]{concept.title}[/yellow] ({concept.domain}, Gen {concept.generation})")
        if Confirm.ask("  [red]Delete?[/red]", default=False):
            if await delete_concept(session, concept.id):
                rprint("  [green]Deleted.[/green]")

async def export_json_logic(file: str, project: str = None):
    async for session in get_async_session():
        stmt = select(Concept).order_by(Concept.generation, Concept.created_at)
        if project:
            stmt = stmt.where(Concept.project == project)
        concepts = (await session.execute(stmt)).scalars().all()
        lineages = (await session.execute(select(Lineage))).scalars().all()
        concept_ids = {c.id for c in concepts}
        relevant_lineages = [l for l in lineages if l.child_id in concept_ids]

        data = {
            "version": "1.0",
            "project": project or "all",
            "exported_at": datetime.now(UTC).isoformat(),
            "concepts": [
                {"id": str(c.id), "project": c.project, "domain": c.domain, "title": c.title,
                 "description": c.description, "generation": c.generation,
                 "fitness_score": c.fitness_score, "tags": c.tags,
                 "created_at": c.created_at.isoformat()}
                for c in concepts
            ],
            "lineages": [
                {"child_id": str(l.child_id), "parent_id": str(l.parent_id), "mutation_type": l.mutation_type}
                for l in relevant_lineages
            ],
        }
        with open(file, "w") as f:
            json.dump(data, f, indent=2)
        rprint(f"  [green]Exported {len(concepts)} concepts to {file}[/green]")

async def import_json_logic(file: str, project: str = None):
    if not os.path.exists(file):
        rprint(f"[red]File not found: {file}[/red]")
        return

    with open(file, "r") as f:
        data = json.load(f)

    concepts_data = data if isinstance(data, list) else data.get("concepts", data.get("seeds", []))
    if not concepts_data:
        rprint("[red]No concepts in file.[/red]")
        return

    count = 0
    async for session in get_async_session():
        for item in concepts_data:
            concept = Concept(
                project=project or item.get("project", "default"),
                domain=item.get("domain", "Unknown"),
                title=item.get("title", "Untitled"),
                description=item.get("description", ""),
                generation=item.get("generation", 0),
                fitness_score=item.get("fitness_score"),
                tags=item.get("tags"),
            )
            await save_concept(session, concept)
            count += 1
    rprint(f"  [green]Imported {count} concepts[/green]")

async def display_graph_logic(project: str = None):
    from rich.tree import Tree
    async for session in get_async_session():
        stmt = select(Concept)
        if project:
            stmt = stmt.where(Concept.project == project)
        stmt = stmt.order_by(Concept.generation, Concept.created_at)
        concepts = (await session.execute(stmt)).scalars().all()

        if not concepts:
            rprint("  [dim]No concepts to graph.[/dim]")
            return

        lineages = (await session.execute(select(Lineage))).scalars().all()
        concept_ids = {c.id for c in concepts}
        concept_map = {c.id: c for c in concepts}

        # Build parent->children map
        children_map: dict = {}
        has_parent = set()
        for l in lineages:
            if l.child_id in concept_ids and l.parent_id in concept_ids:
                children_map.setdefault(l.parent_id, []).append((l.child_id, l.mutation_type))
                has_parent.add(l.child_id)

        roots = [c for c in concepts if c.id not in has_parent]

        root_tree = Tree(f"[bold cyan]{project or 'all'}[/bold cyan] [dim]lineage[/dim]")

        def add_children(tree_node, concept_id):
            for child_id, mut_type in children_map.get(concept_id, []):
                child = concept_map.get(child_id)
                if not child:
                    continue
                icon = {"CROSSOVER": "x", "POINT_MUTATION": "~", "INVERSION": "!", "WILDCARD": "*"}.get(mut_type, "?")
                fit = f" [green]{child.fitness_score:.1f}[/green]" if child.fitness_score else ""
                node = tree_node.add(f"[{icon}] [bold]{child.title}[/bold] [magenta]{child.domain}[/magenta] [dim]Gen {child.generation}[/dim]{fit}")
                add_children(node, child_id)

        for root in roots:
            fit = f" [green]{root.fitness_score:.1f}[/green]" if root.fitness_score else ""
            node = root_tree.add(f"[bold]{root.title}[/bold] [magenta]{root.domain}[/magenta] [dim]Gen {root.generation}[/dim]{fit}")
            add_children(node, root.id)

        console.print(root_tree)
        rprint(f"  [dim]{len(concepts)} concepts, {len([l for l in lineages if l.child_id in concept_ids])} links[/dim]")


async def export_graph_logic(file: str, project: str = None):
    async for session in get_async_session():
        stmt = select(Concept)
        if project:
            stmt = stmt.where(Concept.project == project)
        concepts = (await session.execute(stmt)).scalars().all()
        concept_ids = {c.id for c in concepts}
        lineages = (await session.execute(select(Lineage))).scalars().all()
        relevant_lineages = [l for l in lineages if l.child_id in concept_ids]

        dot_content = ["digraph BlindMind {", '  node [shape=box, fontname="Arial", style=filled, fillcolor="#f0f0f0"];', '  rankdir="LR";']
        for c in concepts:
            label = f"{c.title}\\n(Gen {c.generation})\\nScore: {c.fitness_score or 'N/A'}"
            color = "#e1f5fe" if c.generation == 0 else "#c8e6c9"
            dot_content.append(f'  "{c.id}" [label="{label}", fillcolor="{color}"];')
        for l in relevant_lineages:
            dot_content.append(f'  "{l.parent_id}" -> "{l.child_id}" [label="{l.mutation_type}"];')
        dot_content.append("}")
        with open(file, "w") as f:
            f.write("\n".join(dot_content))
        rprint(f"  [green]Graph exported to {file}[/green]")

async def tree_lineage_logic(concept_id: str):
    from rich.tree import Tree
    async for session in get_async_session():
        all_concepts = (await session.execute(select(Concept))).scalars().all()
        concept = next((c for c in all_concepts if str(c.id).startswith(concept_id)), None)
        if not concept:
            rprint("[red]Not found.[/red]")
            return
        root = Tree(f"[bold green]{concept.title}[/bold green] [dim]Gen {concept.generation}[/dim]")
        async def add_parents(t, cid):
            links = (await session.execute(select(Lineage).where(Lineage.child_id == cid))).scalars().all()
            for l in links:
                p = (await session.execute(select(Concept).where(Concept.id == l.parent_id))).scalars().first()
                if p:
                    icon = {"CROSSOVER": "x", "POINT_MUTATION": "~", "INVERSION": "!", "WILDCARD": "*"}.get(l.mutation_type, "?")
                    node = t.add(f"[cyan]{p.title}[/cyan] [dim]Gen {p.generation} [{icon}][/dim]")
                    await add_parents(node, p.id)
        await add_parents(root, concept.id)
        console.print(root)

async def refine_logic(essay_path: str, style_path: str = None, variants: int = 3, output: str = None):
    from blindmind.refine import EssayRefiner

    if not os.path.exists(essay_path):
        rprint(f"[red]File not found: {essay_path}[/red]")
        return

    with open(essay_path, "r") as f:
        essay_text = f.read()

    essay_name = os.path.splitext(os.path.basename(essay_path))[0]
    project = f"refine:{essay_name}"

    style_guide = None
    if style_path:
        style_guide = EssayRefiner.load_style_guide(style_path)
    else:
        default_style = os.path.expanduser("~/Documents/_Essays/_Articles/_WRITINGSTYLE.md")
        if os.path.exists(default_style):
            style_guide = EssayRefiner.load_style_guide(default_style)
            rprint(f"  [dim]Voice: loaded from {default_style}[/dim]")

    word_count = len(essay_text.split())
    rprint(Panel(
        f"[dim]Essay:[/dim] [bold]{essay_name}[/bold] ({word_count:,} words)\n"
        f"[dim]Project:[/dim] {project}\n"
        f"[dim]Variants per section:[/dim] {variants}",
        title="[bold cyan]Refine[/bold cyan]",
        border_style="cyan",
    ))

    async for session in get_async_session():
        refiner = EssayRefiner(session, project=project, style_guide=style_guide)

        # Step 1: Parse
        with Progress(SpinnerColumn(spinner_name="dots"), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            progress.add_task("[cyan]Parsing essay into sections...", total=None)
            sections = await refiner.parse_essay(essay_text)

        rprint(f"\n  [bold]Section Map[/bold] ({len(sections)} sections)")
        rprint(f"  [dim]{'─' * 50}[/dim]")
        func_icons = {"setup": ".", "escalation": "/", "evidence": "#", "turn": ">", "reflection": "~", "climax": "!", "close": "."}
        for sec in sections:
            icon = func_icons.get(sec["function"], "?")
            preview = sec["content"][:60].replace("\n", " ")
            rprint(f"  [dim]{sec['index']+1:2}[/dim] [{icon}] [magenta]{sec['function']:12}[/magenta] {sec['title']}")
            rprint(f"     [dim]{preview}...[/dim]")
        rprint()

        if not Confirm.ask("  Proceed with refinement?", default=True):
            return

        # Step 2: Seed originals
        await refiner.seed_sections()

        # Step 3: Evolve each section
        selected_content = {i: s["content"] for i, s in enumerate(sections)}
        total_retained = 0

        for sec_idx, sec in enumerate(sections):
            rprint(f"\n  [bold]{'━' * 50}[/bold]")
            rprint(f"  [bold yellow]Section {sec_idx + 1}/{len(sections)}[/bold yellow]: {sec['title']} [dim]({sec['function']})[/dim]")
            rprint(f"  [bold]{'━' * 50}[/bold]")

            # Show original
            console.print(Panel(
                sec["content"],
                title="[dim]Original[/dim]",
                border_style="dim",
                padding=(1, 2),
            ))

            # Generate variants
            with Progress(SpinnerColumn(spinner_name="dots"), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                progress.add_task(f"[cyan]Generating {variants} variants...", total=None)
                section_variants = await refiner.evolve_section(sec_idx, num_variants=variants, selected_content=selected_content)

            if not section_variants:
                rprint("  [red]No variants generated. Keeping original.[/red]")
                continue

            # Display variants
            for vi, (content, approach, critique, mt) in enumerate(section_variants, 1):
                type_icon = {"REWRITE": "~", "TIGHTEN": "-", "AMPLIFY": "+", "INVERSION": "!"}.get(mt, "?")
                type_label = {"REWRITE": "Rewrite", "TIGHTEN": "Tighten", "AMPLIFY": "Amplify", "INVERSION": "Inversion"}.get(mt, mt)

                scores_display = (
                    f"  Voice     {mini_bar(critique.voice_fidelity)} {critique.voice_fidelity}\n"
                    f"  Impact    {mini_bar(critique.emotional_impact)} {critique.emotional_impact}\n"
                    f"  Precision {mini_bar(critique.precision)} {critique.precision}\n"
                    f"  Coherence {mini_bar(critique.coherence)} {critique.coherence}\n"
                    f"  Original  {mini_bar(critique.originality)} {critique.originality}\n"
                    f"\n"
                    f"  [bold]Composite   {score_bar(critique.composite_score)}[/bold]"
                )

                strengths_line = ""
                if critique.strengths:
                    strengths_line = "\n[green]+" + "\n+".join(f" {s}" for s in critique.strengths[:2]) + "[/green]"
                weaknesses_line = ""
                if critique.weaknesses:
                    weaknesses_line = "\n[red]-" + "\n-".join(f" {w}" for w in critique.weaknesses[:2]) + "[/red]"

                border = "green" if critique.composite_score >= 7 else "yellow" if critique.composite_score >= 5 else "red"
                console.print(Panel(
                    f"{content}\n\n"
                    f"[dim italic]{approach}[/dim italic]\n\n"
                    f"{scores_display}"
                    f"{strengths_line}{weaknesses_line}",
                    title=f"[bold][{type_icon}] {type_label}[/bold] [dim]({vi}/{len(section_variants)})[/dim]",
                    border_style=border,
                    padding=(1, 2),
                ))

            # User selection
            rprint(f"  [dim]Enter variant number (1-{len(section_variants)}) to keep, or 0 for original[/dim]")
            pick = Prompt.ask("  [bold]Pick[/bold]", default="0")
            try:
                pick_num = int(pick)
            except ValueError:
                pick_num = 0

            if 1 <= pick_num <= len(section_variants):
                chosen_content, _, chosen_critique, chosen_mt = section_variants[pick_num - 1]
                selected_content[sec_idx] = chosen_content
                await refiner.save_variant(
                    sec_idx, chosen_content,
                    generation=1,
                    fitness=chosen_critique.composite_score,
                    mutation_type=chosen_mt,
                    parent_id=sec["concept_id"],
                )
                total_retained += 1
                rprint(f"  [green]Variant {pick_num} selected[/green] [dim](score: {chosen_critique.composite_score:.1f})[/dim]")
            else:
                rprint("  [dim]Keeping original[/dim]")

        # Step 4: Assemble
        assembled = EssayRefiner.assemble([selected_content[i] for i in range(len(sections))])

        # Step 5: Coherence check
        rprint(f"\n  [bold]{'━' * 50}[/bold]")
        rprint(f"  [bold]Coherence Check[/bold]")

        with Progress(SpinnerColumn(spinner_name="dots"), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            progress.add_task("[cyan]Checking assembled essay...", total=None)
            coherence = await refiner.check_coherence(assembled)

        coherence_display = f"  Overall: {mini_bar(coherence.overall_coherence)} {coherence.overall_coherence}/10"
        if coherence.seams:
            coherence_display += "\n\n  [yellow]Seams:[/yellow]"
            for s in coherence.seams[:3]:
                coherence_display += f"\n  [dim]  {s[:100]}[/dim]"
        if coherence.momentum_breaks:
            coherence_display += "\n\n  [yellow]Momentum breaks:[/yellow]"
            for m in coherence.momentum_breaks[:3]:
                coherence_display += f"\n  [dim]  {m[:100]}[/dim]"
        if coherence.suggestions:
            coherence_display += "\n\n  [green]Fixes:[/green]"
            for sg in coherence.suggestions[:5]:
                coherence_display += f"\n  [dim]  {sg[:100]}[/dim]"

        rprint(coherence_display)

        # Step 6: Save output
        out_path = output or essay_path.replace(".md", "_refined.md")
        with open(out_path, "w") as f:
            f.write(assembled)

        assembled_words = len(assembled.split())
        delta = assembled_words - word_count
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        rprint(f"\n  [green]Saved to {out_path}[/green]")
        rprint(f"  [dim]{assembled_words:,} words ({delta_str} from original) | {total_retained}/{len(sections)} sections refined[/dim]")

def show_settings(project: str = "default"):
    settings_text = (
        f"[cyan]Project[/cyan]      {project}\n"
        f"[cyan]Model[/cyan]        {settings.litellm_model}\n"
        f"[cyan]Temperature[/cyan]  variation={settings.variation_temperature}  critic={settings.critic_temperature}\n"
        f"[cyan]Threshold[/cyan]    {settings.critic_threshold}\n"
        f"[cyan]Rates[/cyan]        crossover={settings.crossover_rate}  mutation={settings.point_mutation_rate}  inversion={settings.inversion_rate}\n"
        f"[cyan]Concurrency[/cyan]  {settings.max_concurrent_calls}\n"
        f"[cyan]Database[/cyan]     {settings.database_url}"
    )
    console.print(Panel(settings_text, title="[bold]Settings[/bold]", border_style="dim"))

# --- CLI Commands ---

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        welcome_banner()
        async def _main():
            global _active_project
            await ensure_setup()

            while True:
                try:
                    # Fetch quick stats for header
                    concept_count = 0
                    async for session in get_async_session():
                        concept_count = (await session.execute(
                            select(func.count(Concept.id)).where(Concept.project == _active_project)
                        )).scalar() or 0

                    rprint(f"\n  [bold cyan]{_active_project}[/bold cyan] [dim]({concept_count} concepts)[/dim]")
                    rprint(f"  [dim]{'─' * 40}[/dim]")
                    rprint(f"   [bold cyan]e[/bold cyan] Evolve    [bold green]s[/bold green] Seed      [bold magenta]R[/bold magenta] Refine")
                    rprint(f"   [bold magenta]l[/bold magenta] List      [bold white]v[/bold white] View      [bold white]f[/bold white] Find")
                    rprint(f"   [bold yellow]t[/bold yellow] Tree      [bold blue]g[/bold blue] Graph     [dim]a[/dim] Stats")
                    rprint(f"   [bold blue]x[/bold blue] Export    [bold blue]i[/bold blue] Import    [dim]r[/dim] Runs")
                    rprint(f"   [bold]p[/bold] Project   [dim]c[/dim] Config    [red]d[/red] Delete")
                    rprint(f"   [dim]q Quit  ? Help[/dim]")

                    choice = Prompt.ask("\n  [bold]>[/bold]", default="e")
                    choice_raw = choice.strip()
                    choice = choice_raw.lower()

                    if choice in ("e", "1", "evolve"):
                        gens = IntPrompt.ask("  Generations", default=1)
                        pop = IntPrompt.ask("  Population", default=5)
                        threshold_str = Prompt.ask(f"  Threshold [dim]({settings.critic_threshold})[/dim]", default="")
                        temp_str = Prompt.ask(f"  Temperature [dim]({settings.variation_temperature})[/dim]", default="")
                        threshold = float(threshold_str) if threshold_str else None
                        temp = float(temp_str) if temp_str else None
                        await run_evolution_logic(gens, pop, threshold=threshold, temperature=temp, project=_active_project)
                    elif choice in ("s", "2", "seed"):
                        domain = Prompt.ask("  Domain [dim](empty to cancel)[/dim]", default="")
                        if not domain.strip():
                            continue
                        title = Prompt.ask("  Title [dim](empty to cancel)[/dim]", default="")
                        if not title.strip():
                            continue
                        desc = Prompt.ask("  Description")
                        tags = Prompt.ask("  Tags [dim](optional)[/dim]", default="")
                        async for session in get_async_session():
                            await save_concept(session, Concept(project=_active_project, domain=domain, title=title, description=desc, generation=0, tags=tags or None))
                        rprint("  [green]Saved.[/green]")
                    elif choice in ("l", "3", "list"):
                        filter_str = Prompt.ask("  [bold]Filter[/bold] [dim](gen:N, domain:X, fit:N, or empty for all)[/dim]", default="")
                        gen_filter = None
                        domain_filter = None
                        fitness_filter = None
                        for part in filter_str.split(","):
                            part = part.strip()
                            if part.startswith("gen:"):
                                try: gen_filter = int(part[4:])
                                except ValueError: pass
                            elif part.startswith("domain:"):
                                domain_filter = part[7:].strip()
                            elif part.startswith("fit:"):
                                try: fitness_filter = float(part[4:])
                                except ValueError: pass
                        await list_concepts_logic(gen_filter, 30, domain=domain_filter, min_fitness=fitness_filter, project=_active_project)
                    elif choice in ("f", "4", "find", "search"):
                        q = Prompt.ask("  Search [dim](empty to cancel)[/dim]", default="")
                        if not q.strip():
                            continue
                        await search_concepts_logic(q, project=_active_project)
                    elif choice in ("v", "5", "view"):
                        cid = await pick_concept_id(_active_project, "View")
                        if cid:
                            await view_concept_logic(cid)
                    elif choice in ("t", "6", "tree"):
                        cid = await pick_concept_id(_active_project, "Tree")
                        if cid:
                            await tree_lineage_logic(cid)
                    elif choice in ("g", "7", "graph"):
                        await display_graph_logic(project=_active_project)
                    elif choice in ("x", "8", "export"):
                        filename = Prompt.ask("  File", default="blindmind_export.json")
                        await export_json_logic(filename, project=_active_project)
                    elif choice in ("i", "9", "import"):
                        filename = Prompt.ask("  File [dim](empty to cancel)[/dim]", default="")
                        if not filename.strip():
                            continue
                        proj_override = Prompt.ask(f"  Project [dim]({_active_project})[/dim]", default=_active_project)
                        await import_json_logic(filename, project=proj_override)
                    elif choice in ("a", "10", "stats"):
                        await stats_logic(project=_active_project)
                    elif choice_raw == "R" or choice == "refine":
                        essay_file = Prompt.ask("  Essay file path")
                        num_variants = IntPrompt.ask("  Variants per section", default=3)
                        await refine_logic(essay_file, variants=num_variants)
                    elif choice in ("r", "11", "runs"):
                        async for session in get_async_session():
                            runs = (await session.execute(
                                select(EvolutionRun).where(EvolutionRun.project == _active_project).order_by(EvolutionRun.created_at.desc()).limit(10)
                            )).scalars().all()
                            if not runs:
                                rprint("  [dim]No runs yet.[/dim]")
                            else:
                                table = Table(border_style="dim", show_edge=False)
                                table.add_column("ID", style="dim", width=8)
                                table.add_column("Status", width=10)
                                table.add_column("Progress", width=10)
                                table.add_column("Kept", justify="right", width=4)
                                table.add_column("When", style="dim", width=12)
                                for r in runs:
                                    sc = {"COMPLETED": "green", "FAILED": "red", "CANCELLED": "yellow"}.get(r.status, "blue")
                                    table.add_row(
                                        str(r.id)[:8],
                                        f"[{sc}]{r.status.lower()}[/{sc}]",
                                        f"{r.current_generation}/{r.total_generations}",
                                        str(r.concepts_retained),
                                        r.created_at.strftime("%m/%d %H:%M"),
                                    )
                                console.print(table)
                    elif choice in ("c", "12", "config", "settings"):
                        show_settings(project=_active_project)
                    elif choice in ("d", "13", "delete"):
                        cid = await pick_concept_id(_active_project, "Delete")
                        if cid:
                            await delete_concept_logic(cid)
                    elif choice in ("p", "14", "project"):
                        async for session in get_async_session():
                            projects = await get_projects(session)
                        if projects:
                            for pi, p in enumerate(projects, 1):
                                marker = " [bold green]*[/bold green]" if p == _active_project else ""
                                rprint(f"    [bold]{pi}[/bold]. {p}{marker}")
                            rprint(f"    [bold]{len(projects)+1}[/bold]. [dim]Create new[/dim]")
                            pick = Prompt.ask(f"  [bold]Switch[/bold] [dim](1-{len(projects)+1}, empty to cancel)[/dim]", default="")
                            if not pick.strip():
                                continue
                            try:
                                pick_num = int(pick)
                                if 1 <= pick_num <= len(projects):
                                    _active_project = projects[pick_num - 1]
                                elif pick_num == len(projects) + 1:
                                    new_name = Prompt.ask("  Project name", default="")
                                    if new_name.strip():
                                        _active_project = new_name.strip()
                                        rprint(f"  [green]Created: {_active_project}[/green]")
                                    else:
                                        continue
                            except ValueError:
                                if pick.strip() in projects:
                                    _active_project = pick.strip()
                        else:
                            new_proj = Prompt.ask("  Project name", default="")
                            if new_proj.strip():
                                _active_project = new_proj.strip()
                        rprint(f"  [green]Active: {_active_project}[/green]")
                    elif choice in ("?", "h", "help"):
                        rprint(Panel(
                            "[bold cyan]e[/bold cyan] Evolve     Run evolutionary generation cycle (crossover, mutation, critique)\n"
                            "[bold green]s[/bold green] Seed       Add a new seed concept manually\n"
                            "[bold magenta]R[/bold magenta] Refine     Evolve essay sections with AI rewriting\n"
                            "[bold magenta]l[/bold magenta] List       Show concepts [dim](supports gen:N, domain:X, fit:N filters)[/dim]\n"
                            "[bold white]v[/bold white] View       Inspect a concept with full details and lineage\n"
                            "[bold white]f[/bold white] Find       Search concepts by keyword in title/description\n"
                            "[bold yellow]t[/bold yellow] Tree       Show ancestry tree for a concept\n"
                            "[bold blue]g[/bold blue] Graph      Display full project lineage as a tree\n"
                            "[dim]a[/dim] Stats      Latent space statistics and generation distribution\n"
                            "[bold blue]x[/bold blue] Export     Export concepts to JSON file\n"
                            "[bold blue]i[/bold blue] Import     Import concepts from JSON file\n"
                            "[dim]r[/dim] Runs       Show recent evolution run history\n"
                            "[bold]p[/bold] Project    Switch or create projects\n"
                            "[dim]c[/dim] Config     Show current settings\n"
                            "[red]d[/red] Delete     Remove a concept\n"
                            "[dim]q[/dim] Quit       Exit the application",
                            title="[bold]Help[/bold]",
                            border_style="dim",
                        ))
                    elif choice in ("q", "0", "quit", "exit"):
                        rprint("  [dim]Done.[/dim]")
                        break
                    else:
                        rprint(f"  [dim]Unknown: '{choice}' (type ? for help)[/dim]")
                except KeyboardInterrupt:
                    rprint("\n  [dim]Press q to exit.[/dim]")
                    continue
        asyncio.run(_main())

@app.command()
def init():
    """Initialize the database."""
    welcome_banner()
    asyncio.run(ensure_setup())

@app.command()
def run(
    generations: int = 1,
    population: int = 5,
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-T"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    project: str = typer.Option("default", "--project", "-p"),
):
    """Start an evolutionary run."""
    asyncio.run(run_evolution_logic(generations, population, threshold=threshold, temperature=temperature, model=model, project=project))

@app.command("list")
def list_cmd(
    generation: Optional[int] = typer.Option(None, "--gen", "-g"),
    limit: int = typer.Option(20, "--limit", "-l"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    min_fitness: Optional[float] = typer.Option(None, "--min-fitness"),
    project: Optional[str] = typer.Option(None, "--project", "-p"),
):
    """List concepts."""
    asyncio.run(list_concepts_logic(generation, limit, domain=domain, min_fitness=min_fitness, project=project))

@app.command()
def search(query: str = typer.Argument(...), domain: Optional[str] = typer.Option(None, "-d"), min_fitness: Optional[float] = typer.Option(None, "--min-fitness"), limit: int = typer.Option(20, "-l"), project: Optional[str] = typer.Option(None, "-p")):
    """Search concepts."""
    asyncio.run(search_concepts_logic(query, domain=domain, min_fitness=min_fitness, limit=limit, project=project))

@app.command()
def view(concept_id: str):
    """View a concept."""
    asyncio.run(view_concept_logic(concept_id))

@app.command()
def delete(concept_id: str):
    """Delete a concept."""
    asyncio.run(delete_concept_logic(concept_id))

@app.command()
def stats(project: Optional[str] = typer.Option(None, "-p")):
    """Latent space statistics."""
    asyncio.run(stats_logic(project=project))

@app.command()
def graph(file: str = "evolution_graph.dot", project: Optional[str] = typer.Option(None, "-p")):
    """Export to Graphviz DOT."""
    asyncio.run(export_graph_logic(file, project=project))

@app.command()
def tree(concept_id: str):
    """Visualize ancestry."""
    asyncio.run(tree_lineage_logic(concept_id))

@app.command("export")
def export_cmd(file: str = typer.Argument("blindmind_export.json"), project: Optional[str] = typer.Option(None, "-p")):
    """Export to JSON."""
    asyncio.run(export_json_logic(file, project=project))

@app.command("import")
def import_cmd(file: str = typer.Argument(...), project: Optional[str] = typer.Option(None, "-p")):
    """Import from JSON."""
    asyncio.run(import_json_logic(file, project=project))

@app.command()
def refine(
    essay: str = typer.Argument(..., help="Path to essay markdown file"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Path to writing style guide"),
    variants: int = typer.Option(3, "--variants", "-v", help="Variants per section"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Refine an essay section by section using evolutionary rewriting."""
    asyncio.run(refine_logic(essay, style_path=style, variants=variants, output=output))

@app.command("settings")
def settings_cmd():
    """Show configuration."""
    show_settings()

@app.command("projects")
def projects_cmd():
    """List all projects."""
    async def _list():
        await init_db()
        async for session in get_async_session():
            projects = await get_projects(session)
            if not projects:
                rprint("  [dim]No projects.[/dim]")
                return
            for p in projects:
                s = await get_stats(session, project=p)
                rprint(f"  [bold]{p}[/bold] [dim]|[/dim] {s['total']} concepts [dim]|[/dim] {s['domains']} domains [dim]|[/dim] gen {s['generations']}")
    asyncio.run(_list())

if __name__ == "__main__":
    app()
