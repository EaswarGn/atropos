import asyncio
import json
import logging
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
from datasets import Dataset, load_dataset
from rich import box
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm.asyncio import tqdm_asyncio
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    EvalHandlingEnum,
    ScoredDataGroup,
    logger,
)
from atroposlib.envs.server_handling.managed_server import SequenceNode

from .chess_env_types import ChessPuzzleItem, MoveExtractionError, RolloutItem
from .configs import ChessEnvConfig
from .curriculum_manager import StatefulCurriculumManager, format_item


class ChessEnv(BaseEnv):
    """Chess environment for reinforcement learning with chess puzzles.

    This environment manages chess puzzle training and evaluation using a curriculum
    manager, integrates with chess engines for move evaluation, and handles scoring
    of model-generated responses based on chess move quality.

    Attributes:
        config (ChessEnvConfig): Configuration for the chess environment.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer for processing text.
    """

    config: ChessEnvConfig
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    def __init__(
        self,
        config: ChessEnvConfig,
        server_configs: List[APIServerConfig],
        testing=False,
        slurm=False,
    ):
        """Initialize the ChessEnv.

        Args:
            config (ChessEnvConfig): Configuration object for the environment.
            server_configs (List[APIServerConfig]): List of API server configurations.
            testing (bool, optional): Whether this is a testing instance. Defaults to False.
            slurm (bool, optional): Whether running on SLURM. Defaults to False.
        """

        super().__init__(config, server_configs, slurm, testing)
        self.eval_metrics = list()
        self.engine = None
        self.eval_semaphore = asyncio.Semaphore(self.config.max_concurrent_evals)
        self.engine_pool: asyncio.Queue[chess.engine.UciProtocol] = asyncio.Queue()
        self.train_state_dict: Dict[str, Any] = None

    @classmethod
    def config_init(self) -> Tuple[ChessEnvConfig, List[APIServerConfig]]:
        """Initialize and return default configuration for ChessEnv.

        Returns:
            Tuple[ChessEnvConfig, List[APIServerConfig]]: A tuple containing the environment
                configuration and list of server configurations.
        """
        env_config = ChessEnvConfig(
            tokenizer_name="codingmonster1234/chess-sft-modelv2",
            group_size=8,
            use_wandb=False,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=100,
            batch_size=8,
            steps_per_eval=20,
            max_token_length=1024 * 2,
            inference_weight=1.0,
            wandb_name="chess_reasoning_rl",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="codingmonster1234/chess-sft-modelv2",
                tokenizer_name="codingmonster1234/chess-sft-modelv2",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
                server_type="vllm",
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """Set up the environment by loading the dataset and starting Stockfish.

        Loads the training and validation datasets, initializes the curriculum manager,
        and starts a pool of Stockfish engines for move evaluation.
        """
        rprint("[yellow]Loading Dataset...[/yellow]")

        self.train = StatefulCurriculumManager(
            curriculum_config=self.config.train_dataset_config
        )
        self.test = self.create_validation_dataset()

        # Print dataset summary
        train_info = f"[bold cyan]Total:[/bold cyan] {len(self.train):,}\n"
        test_info = f"[bold magenta]Total:[/bold magenta] {len(self.test):,}\n"

        summary_panel = Panel(
            Columns(
                [
                    Panel(
                        train_info,
                        title="[bold white]Train Set[/bold white]",
                        border_style="cyan",
                    ),
                    Panel(
                        test_info,
                        title="[bold white]Test Set[/bold white]",
                        border_style="magenta",
                    ),
                ]
            ),
            title="[bold green]Dataset Loaded Successfully[/bold green]",
            expand=False,
        )

        rprint(summary_panel)

        # Initialize Async Stockfish Engine
        try:
            max_engines = self.config.max_concurrent_evals
            # Initialize N independent engines
            for _ in tqdm_asyncio(range(max_engines), desc="Initializing Engine Pool"):
                transport, engine = await chess.engine.popen_uci(
                    self.config.stockfish_path
                )
                # await engine.configure({"Threads": self.config.engine_threads})
                self.engine_pool.put_nowait(engine)

            rprint(
                f"[bold cyan]{max_engines} Stockfish[/bold cyan] engine(s) [green]initialized successfully\n[/green]"
            )
        except Exception as e:
            rprint(
                f"[bold red]Error:[/bold red] Could not start [bold cyan]Stockfish[/bold cyan]. "
                f"Check path: [yellow]{self.config.stockfish_path}[/yellow]\n[red]{e}[/red]"
            )

    def create_validation_dataset(self) -> Dataset:
        """Create a validation dataset by filtering the test set based on the validation dataset config.

        Returns:
            Dataset: The filtered and sampled validation dataset.
        """
        test_set = load_dataset(
            path=self.config.validation_dataset_config.dataset_name,
            split=self.config.validation_dataset_config.split,
        )

        min_rating = self.config.validation_dataset_config.min_rating
        max_rating = self.config.validation_dataset_config.max_rating
        test_set = test_set.filter(
            lambda x: min_rating <= x["rating"] <= max_rating,
            desc=f"Filtering validation dataset by rating ({min_rating}-{max_rating})",
            num_proc=os.cpu_count(),
        )

        total_count = len(test_set)
        max_items = self.config.validation_dataset_config.max_items
        if total_count < max_items:
            raise ValueError(
                f"Validation dataset has only {total_count} items, but max_items is set to {max_items}."
            )

        all_indices = list(range(total_count))
        selected_indices = random.sample(
            all_indices, self.config.validation_dataset_config.max_items
        )
        test_set = test_set.select(selected_indices)
        return test_set

    def save_checkpoint(self, step: int, data=None):
        """Save a checkpoint of the environment state.

        Args:
            step (int): The current training step.
            data (Optional[Dict], optional): Additional data to save. Defaults to None.
        """
        if data is None:
            data = {}
        data["train_state_dict"] = self.train.state_dict()
        super().save_checkpoint(step, data=data)

    def load_checkpoint(self):
        """Load a checkpoint of the environment state.

        Loads the super class checkpoint and restores the training state.
        """
        super().load_checkpoint()

        # self.train_state_dict attribute is set by the super().load_checkpoint() call
        if self.train_state_dict is not None:
            self.train.load_state_dict(self.train_state_dict)

    async def get_next_item(self) -> ChessPuzzleItem:
        """Get the next training item from the dataset.

        Returns:
            ChessPuzzleItem: The next puzzle item for training.

        Raises:
            StopIteration: If the curriculum has reached maximum steps and infinite loop is disabled.
        """
        if self.train.step == self.train._resolve_total_steps():
            logger.info(
                f"Curriculum has reached max steps: {self.train._resolve_total_steps()}."
            )

            if self.train.config.schedule_config.enable_infinite_loop:
                self.train.reset()
                logger.info(
                    f"Curriculum has been reset to the starting difficulty, current step: {self.train.step}"
                )
                logger.info(repr(self.train))

            else:
                raise StopIteration(
                    f"Curriculum has reached max steps: {self.train._resolve_total_steps()}"
                )

        return self.train.get_next_item()

    async def collect_trajectories(
        self, item: ChessPuzzleItem
    ) -> Tuple[ScoredDataGroup, List]:
        """Generate and collect model responses for scoring.

        Args:
            item (ChessPuzzleItem): The puzzle item to generate responses for.

        Returns:
            Tuple[ScoredDataGroup, List]: A tuple containing the scored data group and a list
                of items to backlog (currently empty).
        """

        """prompt = self.tokenizer.apply_chat_template(
            item.prompt, add_generation_prompt=True, tokenize=False
        )"""

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=item.prompt,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
                stop_token_ids=[self.tokenizer.eos_token_id],
            )

            state = managed.get_state()
            nodes: List[SequenceNode] = state["nodes"]

        to_score: List[RolloutItem] = []

        log_file_path = "completions_log.jsonl"
        for node, choice in zip(nodes, chat_completions.choices):
            rollout_item = RolloutItem(
                input_item=item,
                model_completion={
                    "role": "assistant",
                    "content": choice.message.content,
                },
                tokens=node.tokens,
                masks=node.masked_tokens,
                logprobs=node.logprobs,
            )

            # Calculate token counts
            total_tokens = len(rollout_item.tokens)
            # Count how many tokens are actually model generation (not masked as -100)
            generation_tokens = len([1 for mask in rollout_item.masks if mask != -100])

            # Construct the log entry dictionary
            log_entry = {
                "puzzle_id": getattr(item, "uuid", "unknown"),
                "fen": getattr(item, "fen", "unknown"),
                "model_completion": rollout_item.model_completion["content"],
                "total_tokens": total_tokens,
                "generation_tokens": generation_tokens,
            }

            # Append to the JSON lines file
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            print("\n\n")
            print("model completion: ", rollout_item.model_completion)
            print("total tokens: ", len(rollout_item.tokens))
            print("\n\n")
            to_score.append(rollout_item)

        scored_data = await self.score(to_score)
        to_backlog = []

        if scored_data is not None and scored_data["scores"]:
            self.train.update_group(item.uuid, scored_data["scores"])
            self.train.increment_step(1)

        return [scored_data], to_backlog

    async def score(
        self, rollout_group_data: List[RolloutItem]
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Score the generated model responses based on chess move quality.

        Assigns rewards based on format correctness, move legality, and evaluation difference
        compared to the best move. Returns None if scoring fails.

        Args:
            rollout_group_data (List[RolloutItem]): List of rollout items to score.

        Returns:
            Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]: The scored data group
                or None if scoring conditions are not met.
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["inference_logprobs"] = list()

        fen = rollout_group_data[0].input_item.fen
        best_move = rollout_group_data[0].input_item.best_move

        # Calculate the ground truth best move evaluation once per group
        eval_best = await self.throttled_move_eval(fen, best_move)

        random.shuffle(rollout_group_data)

        # Batch evaluate model moves to save time
        eval_tasks = []
        extracted_moves: List[Union[str, MoveExtractionError]] = []

        for item in rollout_group_data:
            prediction = self._extract_prediction(item)
            extracted_moves.append(prediction)

            if isinstance(prediction, MoveExtractionError):
                # Mock task for failed formats
                async def mock_eval():
                    return None

                eval_tasks.append(mock_eval())
            else:
                eval_tasks.append(self.throttled_move_eval(fen, prediction))

        model_evals = await asyncio.gather(*eval_tasks)

        # print(f"\n\nscoring model responses for fen: {fen}, best move: {best_move}, eval of best move: {eval_best}")
        for i, item in enumerate(rollout_group_data):
            prediction = extracted_moves[i]
            eval_chosen = model_evals[i]

            if prediction == MoveExtractionError.MISSING_TAGS:
                # Hard punishment/0 reward if the model fails the `<think>` and `<answer>` formatting
                final_score = self.config.format_fail_reward

            elif prediction == MoveExtractionError.UNPARSABLE_MOVE:
                # 0.01 reward for producing a move not in UCI/SAN format but following <think> and <answer> structure
                # to encourage format correction over complete failure
                final_score = self.config.invalid_move_notation_reward

            elif prediction == MoveExtractionError.ILLEGAL_MOVE:
                # 0.1 reward for producing a move in SAN/UCI format but that is illegal
                # to encourage improvement over time rather than complete discouragement
                final_score = self.config.illegal_move_reward

            else:

                final_score = self.get_stockfish_reward(eval_best, eval_chosen)

            response_tokens = len(
                self.tokenizer.encode(item.model_completion["content"])
            )

            if response_tokens > self.config.max_token_length * 0.95:
                final_score -= self.config.length_penalty_coefficient * (
                    response_tokens / self.config.max_token_length
                )

            tokens = item.tokens
            masks = item.masks
            logprobs = item.logprobs

            if len([1 for mask in masks if mask != -100]) < 10:
                logger.warning(
                    "Trajectory dropped: Active response generation length is less than 10 tokens."
                )
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(final_score)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        # DEBUG CHECK: Log if we failed to collect the complete group size
        if len(scores["tokens"]) < self.config.group_size:
            print(
                f"CRITICAL: Yielded an incomplete group! "
                f"Expected {self.config.group_size}, "
                f"but only packed {len(scores['tokens'])} valid trajectories."
            )

        if not scores["scores"] or all(
            scores["scores"][0] == score for score in scores["scores"]
        ):
            if all(scores["scores"][0] == score for score in scores["scores"]):
                print(f"all scores are the same: {scores['scores'][0]}")
            if not scores["scores"]:
                print("scores aren't right")
            return None

        return scores

    def _extract_prediction(
        self, rollout_item: RolloutItem
    ) -> Union[str, MoveExtractionError]:
        """Extract the chess move from the <answer> tags, ensuring <think> is present.

        Parses the model completion for required <think> and <answer> tags, validates
        the move format and legality.

        Args:
            rollout_item (RolloutItem): The rollout item containing the model completion.

        Returns:
            Union[str, MoveExtractionError]: The extracted move string or an error enum.
        """
        # Require <think> block
        think_match = re.search(
            r"<think>(.*?)</think>",
            rollout_item.model_completion["content"],
            re.DOTALL | re.IGNORECASE,
        )
        # Require <answer> block
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            rollout_item.model_completion["content"],
            re.DOTALL | re.IGNORECASE,
        )

        if not think_match or not answer_match:
            return MoveExtractionError.MISSING_TAGS

        # reasoning_str = think_match.group(1).strip()
        model_move_str = answer_match.group(1).strip()

        # check if model move is legal
        board = chess.Board(rollout_item.input_item.fen)
        try:
            # Try SAN first (e.g., "Nf3")
            move = board.parse_san(model_move_str)
        except ValueError:
            try:
                # If SAN fails, try UCI (e.g., "g1f3")
                move = board.parse_uci(model_move_str)
            except ValueError:
                # If both fail, the format is invalid
                return MoveExtractionError.UNPARSABLE_MOVE

        if move not in board.legal_moves:
            return MoveExtractionError.ILLEGAL_MOVE

        # Return just the cleaned inner text from the answer block
        return model_move_str

    async def _get_move_eval(
        self, engine: chess.engine.UciProtocol, fen: str, move_str: str
    ) -> float:
        """Evaluate a move in centipawns using Stockfish.

        Args:
            engine (chess.engine.UciProtocol): The chess engine to use for evaluation.
            fen (str): The FEN string representing the board position.
            move_str (str): The move string in SAN or UCI format.

        Returns:
            float: The evaluation score in centipawns. Positive if good for the player to move,
                negative if bad. 1000 centipawns = 1 pawn. 10000 for checkmate in favor,
                -10000 for checkmate against.
        """
        board = chess.Board(fen)
        # Try to parse SAN (e.g. Nf3) or UCI (e.g. g1f3)
        try:
            move = board.parse_san(move_str)
        except ValueError:
            move = board.parse_uci(move_str)

        board.push(move)
        # Eval time strictly limited to prevent hanging
        info = await engine.analyse(
            board,
            chess.engine.Limit(
                depth=self.config.stockfish_depth, time=self.config.engine_time_limit
            ),
        )

        # Get score from the perspective of the player who just moved
        score = info["score"].pov(not board.turn).score(mate_score=10000)
        return float(score)

    async def throttled_move_eval(self, fen: str, move_str: str) -> float:
        """Evaluate a move with throttling to limit concurrent evaluations.

        Uses a semaphore to control the number of simultaneous evaluations and manages
        the engine pool.

        Args:
            fen (str): The FEN string representing the board position.
            move_str (str): The move string in SAN or UCI format.

        Returns:
            float: The evaluation score in centipawns.
        """
        async with (
            self.eval_semaphore
        ):  # This line waits if self.config.max_concurrent_evals tasks are already running
            engine = await self.engine_pool.get()
            try:
                return await self._get_move_eval(engine, fen, move_str)
            finally:
                self.engine_pool.put_nowait(engine)

    def get_stockfish_reward(self, eval_best: float, eval_chosen: float) -> float:
        """Calculate a reward based on the difference in evaluation between the best move and the chosen move.

        Args:
            eval_best (float): Evaluation score of the best move in centipawns.
            eval_chosen (float): Evaluation score of the chosen move in centipawns.

        Returns:
            float: The calculated reward between 0 and 1, where 1 is perfect.
        """
        # Calculate the loss in centipawns.
        # We use max(0, ...) because the model shouldn't be penalized
        # if it finds a move even better than the 'best_move' reference.
        eval_diff = max(0.0, eval_best - eval_chosen)

        # Using a Sigmoid-like scaling function:
        # A 'scaling_factor' of 200.0 means a 200cp (2 pawn) blunder
        # results in a score of ~0.5.
        # A 0cp diff always results in 1.0.
        scaling_factor = self.config.reward_scaling_factor
        # final_score = 1.0 / (1.0 + (eval_diff / scaling_factor))
        final_score = 1 / (1 + math.exp(eval_diff / scaling_factor))

        return final_score

    async def rollout_and_score_eval(
        self, test_item: ChessPuzzleItem
    ) -> Dict[str, Union[int, float, None]]:
        """Rollout a single test item and score it for evaluation metrics.

        Generates a model completion, extracts the move, evaluates it, and computes
        various metrics including format correctness, legality, and evaluation reward.

        Args:
            test_item (ChessPuzzleItem): The test puzzle item to evaluate.

        Returns:
            Dict[str, Union[int, float, None]]: A dictionary of evaluation metrics.
        """
        fen = test_item.fen
        best_move = test_item.best_move

        completion = await self.server.chat_completion(
            messages=test_item.prompt,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=self.config.eval_rollout_temperature,
            split="eval",
        )

        rollout_test_item = RolloutItem(
            input_item=test_item,
            model_completion={
                "role": "assistant",
                "content": completion.choices[0].message.content,
            },
        )

        prediction = self._extract_prediction(rollout_test_item)

        length_penalty = 0
        response_tokens = len(
            self.tokenizer.encode(rollout_test_item.model_completion["content"])
        )
        if response_tokens > self.config.max_token_length * 0.95:
            length_penalty = self.config.length_penalty_coefficient * (
                response_tokens / self.config.max_token_length
            )

        if prediction == MoveExtractionError.MISSING_TAGS:
            return {
                "format_correct": 0,
                "eval_reward": 0,
                "eval_diff": 0,
                "perfect_move": 0,
                "is_legal_move": 0,
                "is_valid_move_notation": 0,
                "length_penalty": length_penalty,
                "final_reward": 0.0 - length_penalty,
                "elo_solved": None,
            }
        if prediction == MoveExtractionError.UNPARSABLE_MOVE:
            return {
                "format_correct": 1,
                "eval_reward": 0,
                "eval_diff": 0,
                "perfect_move": 0,
                "is_legal_move": 0,
                "is_valid_move_notation": 0,
                "length_penalty": length_penalty,
                "final_reward": 0.01 - length_penalty,
                "elo_solved": None,
            }
        if prediction == MoveExtractionError.ILLEGAL_MOVE:
            return {
                "format_correct": 1,
                "eval_reward": 0,
                "eval_diff": 0,
                "perfect_move": 0,
                "is_legal_move": 0,
                "is_valid_move_notation": 1,
                "length_penalty": length_penalty,
                "final_reward": 0.1 - length_penalty,
                "elo_solved": None,
            }

        eval_best = await self.throttled_move_eval(fen, best_move)
        eval_chosen = await self.throttled_move_eval(fen, prediction)

        eval_reward = self.get_stockfish_reward(eval_best, eval_chosen)
        eval_diff = max(0.0, eval_best - eval_chosen)
        perfect_move = 1 if eval_chosen >= eval_best else 0

        return {
            "format_correct": 1,
            "eval_reward": eval_reward,
            "eval_diff": eval_diff,
            "perfect_move": perfect_move,
            "is_legal_move": 1,
            "is_valid_move_notation": 1,
            "length_penalty": length_penalty,
            "final_reward": eval_reward - length_penalty,
            "elo_solved": test_item.rating if perfect_move == 1 else None,
        }

    async def evaluate(self, *args, **kwargs):
        """Evaluate the model on the test set and compute metrics.

        Runs rollouts on all test items, scores them, and aggregates evaluation metrics
        such as format accuracy, legal move accuracy, and average evaluation reward.
        """
        eval_tasks = [
            self.rollout_and_score_eval(format_item(test_item))
            for test_item in self.test
        ]
        all_scores = await tqdm_asyncio.gather(
            *eval_tasks,
            desc=f"Evaluating model ({self.server.servers[0].config.model_name}) responses",
        )

        format_corrects = [score["format_correct"] for score in all_scores]
        is_legal_moves = [score["is_legal_move"] for score in all_scores]
        is_valid_move_notations = [
            score["is_valid_move_notation"] for score in all_scores
        ]
        eval_rewards = [score["eval_reward"] for score in all_scores]
        eval_diffs = [score["eval_diff"] for score in all_scores]
        perfect_moves = [score["perfect_move"] for score in all_scores]
        length_penalties = [score["length_penalty"] for score in all_scores]
        final_rewards = [score["final_reward"] for score in all_scores]
        elo_solveds = [
            score["elo_solved"]
            for score in all_scores
            if score["elo_solved"] is not None
        ]

        format_accuracy = (
            sum(format_corrects) / len(format_corrects) if format_corrects else 0
        )
        legal_move_accuracy = (
            sum(is_legal_moves) / len(is_legal_moves) if is_legal_moves else 0
        )
        valid_move_notation_accuracy = (
            sum(is_valid_move_notations) / len(is_valid_move_notations)
            if is_valid_move_notations
            else 0
        )
        avg_eval_diff = sum(eval_diffs) / len(eval_diffs) if eval_diffs else 0
        perfect_move_accuracy = (
            sum(perfect_moves) / len(perfect_moves) if perfect_moves else 0
        )
        avg_length_penalty = (
            sum(length_penalties) / len(length_penalties) if length_penalties else 0
        )
        avg_final_reward = (
            sum(final_rewards) / len(final_rewards) if final_rewards else 0
        )
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0
        avg_elo_solved = sum(elo_solveds) / len(elo_solveds) if elo_solveds else 0

        metrics_to_add = [
            ("eval/format_accuracy", format_accuracy),
            ("eval/legal_move_accuracy", legal_move_accuracy),
            ("eval/valid_move_notation_accuracy", valid_move_notation_accuracy),
            ("eval/perfect_move_accuracy", perfect_move_accuracy),
            ("eval/avg_eval_diff", avg_eval_diff),
            ("eval/avg_length_penalty", avg_length_penalty),
            ("eval/avg_final_reward", avg_final_reward),
            ("eval/avg_eval_reward", avg_eval_reward),
            ("eval/avg_elo_solved", avg_elo_solved),
        ]

        for name, val in metrics_to_add:
            self.eval_metrics.append((name, val))

        self.print_metrics_summary()

    def print_metrics_summary(self):
        """Print the evaluation metrics in a structured Rich table.

        Displays all accumulated evaluation metrics in a formatted table with color coding.
        """
        console = Console()

        table = Table(
            title=f"[bold blue]Evaluation Summary for {self.server.servers[0].config.model_name}[/bold blue]",
            show_header=True,
            header_style="bold magenta",
            box=box.DOUBLE_EDGE,
            show_footer=True,
        )

        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Metric", style="white")
        table.add_column("Value", justify="right")

        for name, value in self.eval_metrics:
            # Split "eval/accuracy" into "eval" and "accuracy"
            category, metric_name = name.split("/") if "/" in name else ("misc", name)

            # Color coding logic
            if isinstance(value, float):
                val_str = f"{value:.4f}"
                # Green for high accuracy, Red for high pawn loss
                color = "green" if (value > 0.8 and "accuracy" in name) else "white"
                if "eval_diff" in name and value > 1.5:
                    color = "red"
            else:
                val_str = str(value)
                color = "white"

            table.add_row(category, metric_name, f"[{color}]{val_str}[/{color}]")

        console.print("\n")
        console.print(table)
        console.print("\n")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to Weights & Biases.

        Aggregates evaluation metrics and training metrics, then logs them to W&B.

        Args:
            wandb_metrics (Optional[Dict], optional): Additional metrics to log. Defaults to None.
        """
        if wandb_metrics is None:
            wandb_metrics = {}

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        wandb_metrics.update(self.train.get_metrics())

        await super().wandb_log(wandb_metrics)


class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI color codes to log messages.

    Colors log levels with high-intensity ANSI colors for better console readability.
    """

    # High Intensity / Bold ANSI Color Codes
    cyan = "\x1b[36;1m"
    white = "\x1b[37;1m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;1m"
    magenta = "\x1b[35;1m"
    reset = "\x1b[0m"

    # Split the format: color the prefix, reset before the message
    # Output: name - LEVELNAME - message
    prefix_fmt = "%(name)s - %(levelname)s"
    suffix_fmt = " - %(message)s"

    def __init__(self):
        """Initialize the ColorFormatter with color-coded format strings for each log level."""
        super().__init__()
        self.FORMATS = {
            logging.DEBUG: self.cyan + self.prefix_fmt + self.reset + self.suffix_fmt,
            logging.INFO: self.white + self.prefix_fmt + self.reset + self.suffix_fmt,
            logging.WARNING: self.yellow
            + self.prefix_fmt
            + self.reset
            + self.suffix_fmt,
            logging.ERROR: self.red + self.prefix_fmt + self.reset + self.suffix_fmt,
            logging.CRITICAL: self.magenta
            + self.prefix_fmt
            + self.reset
            + self.suffix_fmt,
        }

    def format(self, record):
        """Format the log record with appropriate color coding.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with ANSI color codes.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        # We use a temporary formatter to handle the actual string interpolation
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


if __name__ == "__main__":

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(ColorFormatter())
    if not logger.handlers:
        logger.addHandler(console_handler)
    logger.propagate = False

    ChessEnv.cli()
