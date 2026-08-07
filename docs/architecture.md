# Multimodal Knowledge Acquisition Framework

**Version:** v0.2  
**Author:** Atul Prakash  
**Status:** Research Design Draft

---

# Vision

Design a **general-purpose Multimodal Knowledge Acquisition Framework** capable of acquiring structured knowledge from heterogeneous multimodal sources and constructing a Knowledge Graph.

> Food is the **application domain**, while the architecture should remain reusable for other domains.

---

# Design Principles

The framework should satisfy the following principles:

- Modular
- Extensible
- Source Independent
- Modality Independent
- Explainable
- Provenance Aware
- Reusable
- Plug-and-Play

> Adding support for a new source should **not** require changing the downstream pipeline.

---

# High-Level Pipeline

```mermaid
flowchart TD

    subgraph Sources
        BOOK[Books]
        WEB[Website]
        YT[YouTube]
        IG[Instagram]
    end

    subgraph Adapters
        BA[Book Adapter]
        WA[Website Adapter]
        YA[YouTube Adapter]
        IA[Instagram Adapter]
    end

    Package[Content Package]

    Representation[Canonical Representation]

    subgraph Processors
        Text[Text Processor]
        Image[Image Processor]
        Layout[Layout Processor]
        Audio[Speech Processor]
        Video[Video Processor]
    end

    Events[Knowledge Events]

    Fusion[Knowledge Fusion]

    Canonicalization[Entity Resolution & Canonicalization]

    Validation[Ontology Validation]

    Builder[Graph Builder]

    KG[Knowledge Graph]

    BOOK --> BA
    WEB --> WA
    YT --> YA
    IG --> IA

    BA --> Package
    WA --> Package
    YA --> Package
    IA --> Package

    Package --> Representation

    Representation --> Text
    Representation --> Image
    Representation --> Layout
    Representation --> Audio
    Representation --> Video

    Text --> Events
    Image --> Events
    Layout --> Events
    Audio --> Events
    Video --> Events

    Events --> Fusion
    Fusion --> Canonicalization
    Canonicalization --> Validation
    Validation --> Builder
    Builder --> KG
```

---

# 1. Source

A Source is where information originates.

Examples

- Book
- PDF
- Website
- YouTube
- Instagram
- Facebook
- Audio Recording
- Research Paper

A Source represents **where content is obtained**, not the content itself.

---

# 2. Source Adapter

A Source Adapter converts a particular source into a common exchange object.

Examples

- Book Adapter
- Website Adapter
- YouTube Adapter
- Instagram Adapter

Each adapter hides source-specific implementation details.

---

# 3. Content Package

The Content Package is the standard exchange object produced by every Source Adapter.

It represents the collected content in a source-independent manner.

## Responsibilities

- Store collected artifacts
- Store provenance
- Store available modalities
- Store source metadata
- Store relationships

## Non-responsibilities

The Content Package never performs information extraction.

It never stores semantic knowledge.

It is immutable.

---

## Content Package Schema

| Component | Description |
|------------|-------------|
| Identity | Unique package identifier |
| Provenance | Source, URL, retrieval date, author |
| Artifacts | PDF, HTML, Images, Audio, Video |
| Modalities | Text, Image, Audio, Video, Tables, Layout |
| Relationships | External references and links |

---

# 4. Canonical Representation

The Canonical Representation converts raw artifacts into structured representations suitable for downstream processing.

```mermaid
classDiagram

class CanonicalRepresentation {
    <<interface>>
    +Identity
    +Metadata
    +Structure
}

class BookRepresentation {
    +Pages
    +Paragraphs
    +Headings
    +Lists
    +Tables
    +Images
}

class WebsiteRepresentation {
    +Sections
    +Headings
    +Paragraphs
    +Tables
    +Links
    +Images
}

class VideoRepresentation {
    +Transcript
    +Timeline
    +Frames
    +OCRRegions
    +Metadata
}

CanonicalRepresentation <|.. BookRepresentation
CanonicalRepresentation <|.. WebsiteRepresentation
CanonicalRepresentation <|.. VideoRepresentation
```

The Canonical Representation abstracts away source-specific formats while preserving document structure.

---

# 5. Modality Processors

Each processor specializes in a single modality.

Examples

- Text Processor
- Layout Processor
- Image Processor
- Speech Processor
- Video Processor

Processors never communicate directly with each other.

Each processor independently produces Knowledge Events.

---

# 6. Information Extraction

Information Extraction converts Canonical Representations into semantic observations.

Typical tasks include

- Named Entity Recognition
- Relation Extraction
- Event Extraction
- Entity Linking
- Coreference Resolution

---

# 7. Knowledge Events

Knowledge Events represent atomic semantic observations extracted from one modality.

```mermaid
flowchart TD

    KE[Knowledge Events]

    KE --> RECIPE[Recipe Events]
    KE --> ING[Ingredient Events]
    KE --> COOK[Cooking Events]
    KE --> NUT[Nutrition Events]
    KE --> CONTEXT[Context Events]

    RECIPE --> RM[Recipe Mention]

    ING --> IM[Ingredient Mention]
    ING --> QT[Quantity]
    ING --> SUB[Substitution]

    COOK --> CA[Cooking Action]
    COOK --> TM[Timing]
    COOK --> TOOL[Tool Mention]

    NUT --> NC[Nutrition Claim]
    NUT --> WR[Warning]

    CONTEXT --> CN[Cultural Note]
    CONTEXT --> ST[Story]
```

```json
{
    "event_type":"IngredientMention",
    "source":"Book",
    "location":{
        "page":12
    },
    "confidence":0.96,
    "payload":{
        "ingredient":"Toor Dal",
        "quantity":"1 cup"
    }
}
```

Knowledge Events are **observations**, not graph facts.

One Knowledge Event may generate multiple graph facts.

Multiple Knowledge Events may support a single graph fact.

---

# 8. Knowledge Fusion

Knowledge Fusion combines semantically equivalent Knowledge Events originating from multiple modalities and multiple sources.

Fusion performs

- Evidence aggregation
- Conflict detection
- Confidence aggregation
- Event grouping

Fusion computes **global confidence** using multiple observations.

Example

```mermaid
flowchart LR

    A["Speech Event<br/>Add turmeric"] --> F
    B["Description Event<br/>1 tsp turmeric"] --> F
    C["OCR Event<br/>Turmeric"] --> F
    D["Vision Event<br/>Yellow spice detected"] --> F

    F["Knowledge Fusion"]

    F --> E["Unified Observation"]

    E --> T["Ingredient: Turmeric"]
    E --> Q["Quantity: 1 tsp"]
    E --> C1["Confidence: 0.99"]
```

---

# 9. Entity Resolution & Canonicalization

Different sources may refer to the same real-world entity using different names.

Example

```mermaid
flowchart TD

    T1["Toor Dal"]
    T2["Arhar Dal"]
    T3["Tur Dal"]

    CE["Canonical Entity"]

    C["Arhar Dal"]

    T1 --> CE
    T2 --> CE
    T3 --> CE

    CE --> C
```

This stage creates canonical identifiers before graph construction.

---

# 10. Ontology Validation

Candidate facts are validated against the domain ontology.

Examples

```mermaid
flowchart LR

    A1[Recipe]
    B1["hasIngredient"]
    C1[Toor Dal]

    A1 --> B1 --> C1

    C1 --> V1{{Ontology Validation}}
    V1 -->|✓| KG1[(Knowledge Graph)]

    A2[Recipe]
    B2["hasIngredient"]
    C2[Pressure Cooker]

    A2 --> B2 --> C2

    C2 --> V2{{Ontology Validation}}
    V2 -->|✗| ERR[Constraint Violation]
```

Ontology validation ensures semantic consistency before graph construction.

---

# 11. Graph Builder

Graph Builder converts validated knowledge into the target graph representation.

```mermaid
flowchart TD

    Builder[Graph Builder]

    subgraph Targets["Graph Targets"]
        RDF[RDF]
        PG[Property Graph]
    end

    subgraph Implementations["Example Implementations"]
        Jena[Apache Jena]
        Neo4j[Neo4j]
        GraphDB[GraphDB]
    end

    Builder --> RDF
    Builder --> PG

    RDF --> Jena
    RDF --> GraphDB

    PG --> Neo4j
```

Changing graph technology should not affect upstream components.

---

# 12. Knowledge Graph

The Knowledge Graph stores validated domain knowledge together with provenance and supporting evidence.

Graph nodes and relationships should remain traceable back to their originating Knowledge Events.

---
