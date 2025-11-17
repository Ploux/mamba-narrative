## Narrative State-Space Transformers: Implementation Strategy

**Updated 17 November 2025**

### Core Technical Insight

The innovation extends beyond using SSMs to designing them specifically for narrative structure. While recent Memory-Augmented Transformer work explores SSMs with different temporal dynamics, this approach employs **parallel SSM tracks with learned, narrative-specific discretization**.

### The Architecture Design

**Base Model Choice:** Mamba's selective SSM (S6) serves as the foundation, with significant modifications:

1. **Triple-Track SSM Block**

    - Three parallel SSMs replace Mamba's single SSM per layer
    - Each track processes the full sequence with different state transition matrices (A) and discretization parameters (Δ)
    - The discretization step Δ controls the "memory" of each track:
        - Character track: Large Δ (slow forgetting)
        - Plot track: Medium Δ
        - Scene track: Small Δ (fast forgetting)
2. **Narrative-Aware Selection Mechanism**

    - Mamba uses input-dependent selection for memory retention
    - A "narrative classifier" routes information to the appropriate track
    - This classifier learns to identify: character descriptions, plot events, dialogue, scene changes
    - Learned soft routing (not hard assignment) ensures gradient flow
3. **Cross-Track Information Flow**

    - Cross-track attention occurs every N layers
    - The fast scene track can query the slow character track when needed
    - This prevents information silos while maintaining temporal specialization

### Key Differentiators from Vanilla Mamba

**Mamba's Design:** Single SSM with input-dependent selection, optimized for general sequences

**Proposed Design:** Multiple parallel SSMs with narrative-aware routing, optimized for multi-scale storytelling

**Mathematical Framework:**

- Standard SSM: h'(t) = Ah(t) + Bx(t)
- Proposed approach: Three parallel hidden states with coupled dynamics
    - h'_char(t) = A_char·h_char(t) + B_char·x(t) + α·cross_attention(h_plot, h_scene)
    - Different A matrices learned for different narrative timescales
    - Coupling term α allows information exchange

### Implementation Roadmap

**Phase 1: Baseline Mamba**

1. Establish Mamba-130M on available hardware
2. Train on narrative dataset to establish baseline
3. Profile memory usage and generation quality
4. Create reference implementation for modification

**Phase 2: Multi-Track Architecture**

1. Fork the Mamba codebase
2. Replace single SSM blocks with triple-track blocks
3. Set fixed discretization ratios (1:10:100 for scene:plot:character)
4. Train on same data, measure character consistency improvements

**Phase 3: Narrative-Aware Routing**

1. Add the narrative classifier module
2. Implement simple features: punctuation patterns, entity detection, dialogue markers
3. Use Gumbel-softmax for differentiable routing during training
4. The router should learn that "John walked" routes to scene track while "John was brave" routes to character track

**Phase 4: Cross-Track Attention**

1. Add sparse cross-attention every 4th layer
2. Implement learned gates to determine when tracks exchange information
3. This enables checking character track when generating dialogue

### Training Considerations

**Dataset Strategy:**

- **Project Gutenberg**: Public domain novels with consistent literary quality
    - Focus on 19th-20th century fiction (Austen, Dickens, Twain, etc.)
    - Well-edited, professionally published works
    - Clear narrative structure and character development
- **BookCorpus**: If available, contains over 11,000 published books
    - Modern fiction across multiple genres
    - Used for training BERT, enabling baseline comparisons
- **PG-19**: Curated subset of Project Gutenberg for language modeling
    - Books published before 1919
    - Filtered for quality and length
    - Includes metadata for genre classification

**Specialized Validation Sets:**

- Character consistency tests (same character across 5k words)
- Plot coherence tests (Chekhov's gun - mentioned items should return)
- Scene continuity tests (spatial consistency within scenes)

**Loss Function Design:**

- Primary: Standard language modeling loss
- Auxiliary: Narrative consistency loss
    - Extract character states at different positions using the character track
    - Penalize contradictions (L2 distance between same-character representations)
    - Reward consistent plot progression (increasing complexity in plot track)

**Hyperparameter Focus:**

- Discretization ratios (how much slower is character vs scene track)
- Router temperature (how sharp is the track assignment)
- Cross-attention frequency (every N layers)
- State dimension per track (character track may require more dimensions)

### Evaluation Strategy

**Automated Metrics:**

- Character name consistency across documents
- State drift measurement - quantifying character representation changes over time
- Plot thread completion rate (mentioned elements that get resolved)

**Human Evaluation Tasks:**

- Provide readers with first 2k words, generate next 2k, assess consistency
- Character knowledge quiz testing recall of early-established facts
- Comparison against same-size Mamba and GPT-2 baselines

### Theoretical Justification

Recent discussions at NeurIPS 2024 highlighted that SSMs show promise across diverse applications beyond efficiency gains. This contribution demonstrates that for narrative specifically, multiple timescales are essential rather than merely beneficial.

The biological inspiration provides strong motivation: humans process stories at multiple timescales simultaneously. Readers track immediate action while maintaining stable character models and overarching plot understanding. This architecture explicitly models this cognitive process.

### Practical Starting Point

1. Establish Mamba-130M baseline, begin training on Project Gutenberg subset
2. Implement simplified two-track version (fast/slow) with fixed discretization
3. Add third track and basic routing mechanism
4. Implement cross-track attention and evaluate performance
5. Fine-tune hyperparameters based on validation performance

Each component can be tested independently, reducing implementation risk. The incremental approach allows for debugging at each stage while maintaining scientific rigor.

The contribution demonstrates that **narrative-specific architectural inductive bias outperforms scale** for story generation. Even matching a 7B standard model's narrative performance with a 1B parameter model would constitute a significant contribution to the field.

