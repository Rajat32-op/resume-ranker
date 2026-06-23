from .dimension_schema import Dimension


def build_dimension_templates() -> list[Dimension]:
    """
    Returns the fixed dimensions and their semantic anchors.
    No weights are assigned yet.
    """

    dimensions = [

        Dimension(
            name="TECHNICAL_SKILLS",
            description="Core AI and software engineering skills.",
            weight=0.0,
            anchors=[
                "python",
                "machine learning",
                "deep learning",
                "llm",
                "fine tuning",
                "lora",
                "qlora",
                "peft",
                "transformers",
                "distributed systems",
                "inference optimization"
            ]
        ),

        Dimension(
            name="RANKING_RETRIEVAL",
            description="Search, retrieval and recommendation systems.",
            weight=0.0,
            anchors=[
                "retrieval",
                "ranking",
                "search",
                "recommendation system",
                "embeddings",
                "vector database",
                "hybrid retrieval",
                "dense retrieval",
                "bm25",
                "faiss",
                "pinecone",
                "qdrant",
                "weaviate",
                "milvus",
                "elasticsearch",
                "opensearch",
                "reranking"
            ]
        ),

        Dimension(
            name="PRODUCTION_EXPERIENCE",
            description="Production deployment and real-user experience.",
            weight=0.0,
            anchors=[
                "production",
                "deployed",
                "real users",
                "scale",
                "operational",
                "deployment",
                "index refresh",
                "retrieval-quality regression",
                "production code"
            ]
        ),

        Dimension(
            name="EVALUATION_FRAMEWORKS",
            description="Evaluation and experimentation.",
            weight=0.0,
            anchors=[
                "evaluation",
                "ndcg",
                "mrr",
                "map",
                "offline benchmark",
                "ab testing",
                "a/b testing",
                "metrics",
                "online evaluation",
                "correlation"
            ]
        ),

        Dimension(
            name="PRODUCT_MINDSET",
            description="Startup mindset and shipping quickly.",
            weight=0.0,
            anchors=[
                "ship a working",
                "working system",
                "scrappy",
                "startup",
                "learn from real users",
                "move fast",
                "iterate quickly",
                "high-leverage",
                "async-first",
                "disagree openly",
            ]
        ),

        Dimension(
            name="DOMAIN_EXPERIENCE",
            description="Marketplace and recruiting domain knowledge.",
            weight=0.0,
            anchors=[
                "hr tech",
                "recruiting",
                "marketplace",
                "candidate matching",
                "talent intelligence"
            ]
        ),

        Dimension(
            name="LEADERSHIP",
            description="Mentoring and architectural ownership.",
            weight=0.0,
            anchors=[
                "mentor",
                "mentoring",
                "lead",
                "architecture",
                "own",
                "ownership",
                "growing team"
            ]
        ),

        Dimension(
            name="LOCATION",
            description="Location and relocation requirements.",
            weight=0.0,
            anchors=[
                "pune",
                "noida",
                "hyderabad",
                "mumbai",
                "delhi",
                "relocation"
            ]
        ),

        Dimension(
            name="BEHAVIORAL_SIGNALS",
            description="Availability and engagement.",
            weight=0.0,
            anchors=[
                "active",
                "job market",
                "response rate",
                "notice period",
                "open to work"
            ]
        )

    ]

    return dimensions