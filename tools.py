"""
Weaviate MCP Tools for Blog Post Database
==========================================
Version: 2.0.0 (FastMCP 2 with decorator-based registration)
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import os
from typing import Any, Dict, Optional


STYLE_GUIDE_PATH = "./sanjay_sahay_style.txt"
MIN_POSTS_FOR_PATTERN = 5

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import Filter, MetadataQuery, Sort
from weaviate.auth import AuthApiKey

# mcp will be set by mcp_server.py before tools are registered
# This avoids circular imports
mcp = None

WEAVIATE_CLOUD_URL = os.getenv("WEAVIATE_CLOUD_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Add validation
if not WEAVIATE_CLOUD_URL or not WEAVIATE_API_KEY:
    raise RuntimeError(
        "Missing Weaviate Cloud configuration. "
        "Please set WEAVIATE_CLOUD_URL and WEAVIATE_API_KEY environment variables."
    )

def register_tools():
    """Register all tools with the mcp instance.
    This is called by mcp_server.py after setting mcp."""
    if mcp is None:
        raise RuntimeError("mcp instance not set. Cannot register tools.")
    
    # Apply decorators to all tool functions
    mcp.tool()(search_posts_hybrid)
    mcp.tool()(search_by_date_range)
    mcp.tool()(get_post_by_id)
    mcp.tool()(get_posts_batch)
    mcp.tool()(search_posts_by_topic)
    mcp.tool()(get_topic_statistics)
    mcp.tool()(find_similar_posts)
    mcp.tool()(search_by_keyword)
    mcp.tool()(list_all_topics)
    mcp.tool()(get_recent_posts)
    mcp.tool()(aggregate_posts)
    mcp.tool()(search_chunks)

    mcp.tool()(get_posts_for_daily)
    mcp.tool()(add_writing_pattern)
    mcp.tool()(get_style_guide)

# -----------------------------
# 1. Lazy Load Embedding Model
# -----------------------------
EMBEDDING_MODEL = None
MODEL_DIMENSION = 768
_model_load_attempted = False


from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

def _load_embedding_model():
    """Lazy load embedding model using transformers (lightweight)"""
    global EMBEDDING_MODEL, _model_load_attempted
    
    if EMBEDDING_MODEL is not None:
        return EMBEDDING_MODEL
    
    if _model_load_attempted:
        raise RuntimeError("Failed to load embedding model previously.")
    
    _model_load_attempted = True
    
    try:
        print("📦 Loading lightweight embedding model...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        EMBEDDING_MODEL = {"tokenizer": tokenizer, "model": model}
        print("✅ Embedding model loaded successfully.")
        return EMBEDDING_MODEL
    except Exception as e:
        print(f"🚨 ERROR: Could not load model. Error: {e}")
        EMBEDDING_MODEL = None
        raise RuntimeError(f"Failed to load embedding model: {e}")


def get_embedding_for_query(text: str) -> list:
    """Generate embedding using lightweight model"""
    model_dict = _load_embedding_model()
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].tolist()


# -----------------------------
# Weaviate Connection
# -----------------------------


def get_weaviate_client():
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_CLOUD_URL,
            auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY),
            skip_init_checks=True
        )
        return client
    except Exception as e:
        print(f"ERROR: Failed to connect to Weaviate Cloud: {e}")
        raise RuntimeError(f"Weaviate Cloud connection failed: {e}")


# -----------------------------
# JSON Safe Date Helpers (YYYY-MM-DD)
# -----------------------------


def _format_date(date_input: Any) -> Optional[str]:
    """
    Safely converts a datetime object OR an ISO date string (from Weaviate)
    to 'YYYY-MM-DD' format.
    """
    dt = None
    if isinstance(date_input, datetime):
        dt = date_input
    elif isinstance(date_input, str):
        try:
            dt = datetime.fromisoformat(date_input.replace("Z", "+00:00"))
        except ValueError:
            return date_input

    if dt:
        return dt.strftime("%Y-%m-%d")
    return None


def _parse_date_input(date_str: str) -> datetime:
    """
    Parses a 'YYYY-MM-DD' string into a datetime object.
    """
    return datetime.strptime(date_str, "%Y-%m-%d")


# =============================
# TOOL 1: HYBRID SEARCH
# =============================

def search_posts_hybrid(
    query: str,
    limit: int = 10,
    alpha: float = 0.7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic_filter: Optional[str] = None,
    include_scores: bool = True,
) -> Dict[str, Any]:
    """Advanced hybrid search combining semantic and keyword matching."""
    client = get_weaviate_client()
    try:
        try:
            query_vector = get_embedding_for_query(query)
        except Exception as e:
            return {"success": False, "error": f"Failed to generate query vector: {e}", "query": query}

        post_collection = client.collections.get("Post")
        filters = []
        
        if start_date and end_date:
            start_dt = _parse_date_input(start_date).replace(hour=0, minute=0, second=0)
            end_dt = _parse_date_input(end_date).replace(hour=23, minute=59, second=59)
            start_iso = start_dt.isoformat() + "Z"
            end_iso = end_dt.isoformat() + "Z"
            filters.append(
                Filter.by_property("post_date").greater_or_equal(start_iso)
                & Filter.by_property("post_date").less_or_equal(end_iso)
            )

        if topic_filter:
            filters.append(Filter.by_property("final_topic").equal(topic_filter))

        combined_filter = (
            filters[0] if len(filters) == 1 
            else (filters[0] & filters[1] if len(filters) == 2 else None)
        )

        results = post_collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            filters=combined_filter,
            query_properties=["post_content", "post_title", "final_topic"],
            return_metadata=MetadataQuery(score=True) if include_scores else None,
            return_properties=[
                "post_number", "post_title", "post_content", "final_topic",
                "topic_confidence", "post_date", "secondary_topics",
            ],
        )

        formatted_results = []
        for obj in results.objects:
            result = {
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "content": obj.properties.get("post_content", "")[:300] + "...",
                "primary_topic": obj.properties.get("final_topic"),
                "topic_confidence": obj.properties.get("topic_confidence"),
                "date": _format_date(obj.properties.get("post_date")),
                "secondary_topics": obj.properties.get("secondary_topics"),
            }
            if include_scores and hasattr(obj.metadata, "score"):
                result["relevance_score"] = obj.metadata.score
            formatted_results.append(result)

        return {
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "search_params": {"alpha": alpha, "limit": limit, "filters_applied": bool(filters)},
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}
    finally:
        if client:
            client.close()

# =============================
# TOOL 2: DATE RANGE SEARCH
# =============================

def search_by_date_range(
    start_date: str,
    end_date: str,
    limit: int = 20,
    topic_filter: Optional[str] = None,
    sort_order: str = "desc",
) -> Dict[str, Any]:
    """Search posts within a specific date range."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")

        # FIX: Simple string concatenation like your friend's code
        start_iso = start_date + "T00:00:00Z"
        end_iso = end_date + "T23:59:59Z"

        date_filter = (
            Filter.by_property("post_date").greater_or_equal(start_iso)
            & Filter.by_property("post_date").less_or_equal(end_iso)
        )

        if topic_filter:
            date_filter = date_filter & Filter.by_property("final_topic").equal(topic_filter)

        results = post_collection.query.fetch_objects(
            filters=date_filter,
            limit=limit,
            return_properties=[
                "post_number", "post_title", "post_date", "final_topic",
                "topic_confidence", "post_content", "secondary_topics",
            ],
        )

        formatted_results = []
        for obj in results.objects:
            raw_date = obj.properties.get("post_date")
            formatted_results.append({
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "date": _format_date(raw_date),
                "primary_topic": obj.properties.get("final_topic"),
                "topic_confidence": obj.properties.get("topic_confidence"),
                "preview": obj.properties.get("post_content", "")[:150] + "...",
                "secondary_topics": obj.properties.get("secondary_topics"),
            })

        # Sort by post_number or date if needed
        if sort_order == "desc":
            formatted_results.sort(key=lambda x: x["post_number"], reverse=True)
        else:
            formatted_results.sort(key=lambda x: x["post_number"])

        return {
            "success": True,
            "date_range": f"{start_date} to {end_date}",
            "total_results": len(formatted_results),
            "topic_filter": topic_filter,
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 3: GET POST BY ID
# =============================

def get_post_by_id(post_number: int) -> Dict[str, Any]:
    """Retrieve a complete single post by its ID."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        results = post_collection.query.fetch_objects(
            filters=Filter.by_property("post_number").equal(post_number),
            limit=1,
            return_properties=[
                "post_number", "post_title", "post_content", "post_date",
                "final_topic", "topic_confidence", "secondary_topics",
                "sentence_level_explanation", "word_level_explanation",
            ],
        )
        if not results.objects:
            return {"success": False, "error": f"Post #{post_number} not found"}
        
        obj = results.objects[0]
        return {
            "success": True,
            "post": {
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "content": obj.properties.get("post_content"),
                "date": _format_date(obj.properties.get("post_date")),
                "primary_topic": obj.properties.get("final_topic"),
                "topic_confidence": obj.properties.get("topic_confidence"),
                "secondary_topics": obj.properties.get("secondary_topics"),
                "sentence_level_explanation": obj.properties.get("sentence_level_explanation"),
                "word_level_explanation": obj.properties.get("word_level_explanation"),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 4: BATCH POST RETRIEVAL
# =============================

def get_posts_batch(
    post_numbers: List[int], include_content: bool = True
) -> Dict[str, Any]:
    """Batch retrieve multiple posts by their IDs."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")

        properties_list = [
            "post_number", "post_title", "post_date", "final_topic",
            "topic_confidence", "secondary_topics"
        ]
        if include_content:
            properties_list.append("post_content")

        results = post_collection.query.fetch_objects(
            filters=Filter.by_property("post_number").contains_any(post_numbers),
            limit=len(post_numbers),
            return_properties=properties_list,
        )

        found_posts, found_ids = [], set()
        for obj in results.objects:
            post_id = obj.properties.get("post_number")
            found_ids.add(post_id)
            post_data = {
                "post_number": post_id,
                "title": obj.properties.get("post_title"),
                "date": _format_date(obj.properties.get("post_date")),
                "primary_topic": obj.properties.get("final_topic"),
                "topic_confidence": obj.properties.get("topic_confidence"),
                "secondary_topics": obj.properties.get("secondary_topics"),
            }
            if include_content and "post_content" in obj.properties:
                post_data["content"] = obj.properties.get("post_content")
            found_posts.append(post_data)
        
        missing = [pid for pid in post_numbers if pid not in found_ids]
        return {
            "success": True,
            "requested_count": len(post_numbers),
            "found_count": len(found_posts),
            "missing_posts": missing,
            "posts": found_posts,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 5: TOPIC SEARCH
# =============================

def search_posts_by_topic(
    topic_name: str,
    limit: int = 15,
    fuzzy: bool = True,
    include_secondary: bool = False,
) -> Dict[str, Any]:
    """Search posts by topic name with optional fuzzy matching."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        
        if fuzzy:
            all_posts = post_collection.query.fetch_objects(
                limit=5000,
                return_properties=["final_topic", "secondary_topics"],
            )
            matching_topics, matched_topic_names = set(), set()
            search_lower = topic_name.lower()
            
            for post in all_posts.objects:
                final_topic = post.properties.get("final_topic", "") or ""
                secondary_topics = post.properties.get("secondary_topics", "") or ""
                
                if (search_lower in final_topic.lower() or 
                    search_lower in secondary_topics.lower()):
                    matching_topics.add(final_topic)
                    matched_topic_names.add(final_topic)
            
            if not matching_topics:
                return {"success": False, "error": "No matching topics found"}
            
            if include_secondary:
                filters = None
                for topic in matching_topics:
                    topic_filter = (
                        Filter.by_property("final_topic").equal(topic) |
                        Filter.by_property("secondary_topics").like(topic)
                    )
                    filters = topic_filter if filters is None else (filters | topic_filter)
            else:
                filters = None
                for topic in matching_topics:
                    topic_filter = Filter.by_property("final_topic").equal(topic)
                    filters = topic_filter if filters is None else (filters | topic_filter)
            
            results = post_collection.query.fetch_objects(
                filters=filters,
                limit=limit,
                return_properties=[
                    "post_number", "post_title", "post_date", "final_topic",
                    "topic_confidence", "post_content", "secondary_topics",
                ],
            )
            matched_topics_list = list(matched_topic_names)
        else:
            if include_secondary:
                results = post_collection.query.bm25(
                    query=topic_name,
                    query_properties=["final_topic", "secondary_topics"],
                    limit=limit,
                    return_properties=[
                        "post_number", "post_title", "post_date", "final_topic",
                        "topic_confidence", "post_content", "secondary_topics",
                    ],
                )
            else:
                results = post_collection.query.fetch_objects(
                    filters=Filter.by_property("final_topic").equal(topic_name),
                    limit=limit,
                    return_properties=[
                        "post_number", "post_title", "post_date", "final_topic",
                        "topic_confidence", "post_content", "secondary_topics",
                    ],
                )
            matched_topics_list = [topic_name]

        formatted_results = []
        for obj in results.objects:
            result_data = {
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "date": _format_date(obj.properties.get("post_date")),
                "primary_topic": obj.properties.get("final_topic"),
                "topic_confidence": obj.properties.get("topic_confidence"),
                "preview": obj.properties.get("post_content", "")[:150] + "...",
                "secondary_topics": obj.properties.get("secondary_topics"),
            }
            if include_secondary:
                result_data["is_secondary"] = (
                    obj.properties.get("final_topic") != topic_name
                )
            formatted_results.append(result_data)

        response = {
            "success": True,
            "topic_search": topic_name,
            "fuzzy_match": fuzzy,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }
        if fuzzy:
            response["matched_topics"] = matched_topics_list
        return response
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 6: TOPIC STATISTICS
# =============================

def get_topic_statistics(
    top_n: int = 15, include_distribution: bool = True
) -> Dict[str, Any]:
    """Get comprehensive statistics on topic distribution."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        all_posts = post_collection.query.fetch_objects(
            limit=10000,
            return_properties=[
                "final_topic", "topic_confidence", "secondary_topics",
            ],
        )
        
        topic_counts = defaultdict(int)
        multi_label_count = 0
        topic_count_distribution = defaultdict(int)

        for item in all_posts.objects:
            topic_name = item.properties.get("final_topic", "Unknown")
            if topic_name:
                topic_counts[topic_name] += 1
            
            if item.properties.get("secondary_topics"):
                multi_label_count += 1
            
            if include_distribution:
                secondary_topics = item.properties.get("secondary_topics", "")
                num_topics = 1 + (len(secondary_topics.split(",")) if secondary_topics else 0)
                topic_label = (
                    f"{num_topics}_topic{'s' if num_topics != 1 else ''}"
                    if num_topics < 4 else "4+_topics"
                )
                topic_count_distribution[topic_label] += 1

        sorted_topics = sorted(
            topic_counts.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        
        total_posts_count = len(all_posts.objects)
        topic_breakdown = [
            {
                "topic_name": name,
                "post_count": count,
                "percentage": round((count / total_posts_count) * 100, 2)
                if total_posts_count > 0 else 0,
            }
            for name, count in sorted_topics
        ]

        response = {
            "success": True,
            "statistics": {
                "total_posts": total_posts_count,
                "unique_topics": len(topic_counts),
                "multi_label_posts": multi_label_count,
                "multi_label_percentage": round(
                    (multi_label_count / total_posts_count) * 100, 2
                ) if total_posts_count > 0 else 0,
                "avg_posts_per_topic": round(total_posts_count / len(topic_counts), 2)
                if topic_counts else 0,
            },
            "top_topics": topic_breakdown,
        }
        if include_distribution:
            response["distribution"] = dict(topic_count_distribution)
        return response
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 7: SEMANTIC SIMILARITY
# =============================

def find_similar_posts(
    post_number: int, limit: int = 5, min_similarity: float = 0.7
) -> Dict[str, Any]:
    """Find semantically similar posts to a reference post."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        reference = post_collection.query.fetch_objects(
            filters=Filter.by_property("post_number").equal(post_number),
            limit=1,
            return_properties=[
                "post_number", "post_title", "final_topic",
                "topic_confidence", "avg_embedding",
            ],
        )
        if not reference.objects:
            return {"success": False, "error": f"Reference post #{post_number} not found"}
        
        ref_obj = reference.objects[0]
        ref_embedding = ref_obj.properties.get("avg_embedding")
        if not ref_embedding:
            return {"success": False, "error": f"Post #{post_number} has no embedding"}

        similar = post_collection.query.near_vector(
            near_vector=ref_embedding,
            limit=limit + 1,
            return_metadata=MetadataQuery(distance=True),
            return_properties=[
                "post_number", "post_title", "final_topic",
                "topic_confidence", "post_content",
            ],
        )
        
        similar_posts = []
        for obj in similar.objects:
            if obj.properties.get("post_number") == post_number:
                continue
            similarity = (
                1 - obj.metadata.distance if hasattr(obj.metadata, "distance") else 0
            )
            if similarity >= min_similarity:
                similar_posts.append({
                    "post_number": obj.properties.get("post_number"),
                    "title": obj.properties.get("post_title"),
                    "primary_topic": obj.properties.get("final_topic"),
                    "topic_confidence": obj.properties.get("topic_confidence"),
                    "similarity_score": round(similarity, 4),
                    "preview": obj.properties.get("post_content", "")[:150] + "...",
                })
        
        return {
            "success": True,
            "reference_post": {
                "post_number": ref_obj.properties.get("post_number"),
                "title": ref_obj.properties.get("post_title"),
                "primary_topic": ref_obj.properties.get("final_topic"),
                "topic_confidence": ref_obj.properties.get("topic_confidence"),
            },
            "similar_posts": similar_posts[:limit],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 8: KEYWORD SEARCH
# =============================

def search_by_keyword(
    keyword: str,
    search_in: List[str] = None,
    limit: int = 10,
    exact_match: bool = False,
) -> Dict[str, Any]:
    """Pure keyword search using BM25 algorithm."""
    if search_in is None:
        search_in = ["content", "title"]

    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        field_map = {
            "content": "post_content",
            "title": "post_title",
            "topic": "final_topic",
        }
        query_properties = [
            field_map[field] for field in search_in if field in field_map
        ]

        results = post_collection.query.bm25(
            query=keyword,
            query_properties=query_properties,
            limit=limit,
            return_properties=[
                "post_number", "post_title", "post_content",
                "final_topic", "topic_confidence", "post_date",
            ],
        )

        formatted_results = []
        for obj in results.objects:
            content = obj.properties.get("post_content", "")
            match_idx = content.lower().find(keyword.lower())
            match_context = (
                "..." + content[
                    max(0, match_idx - 50): min(
                        len(content), match_idx + len(keyword) + 50
                    )
                ] + "..."
                if match_idx != -1
                else content[:150] + "..."
            )
            formatted_results.append({
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "primary_topic": obj.properties.get("final_topic"),
                "topic_confidence": obj.properties.get("topic_confidence"),
                "date": _format_date(obj.properties.get("post_date")),
                "preview": content[:200] + "...",
                "match_context": match_context,
            })
        
        return {
            "success": True,
            "keyword": keyword,
            "search_fields": search_in,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 9: LIST ALL TOPICS
# =============================

def list_all_topics(sort_by: str = "count", min_posts: int = 1) -> Dict[str, Any]:
    """List all available topics with post counts."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        all_posts = post_collection.query.fetch_objects(
            limit=10000, return_properties=["final_topic"]
        )
        topic_counts = defaultdict(int)

        for item in all_posts.objects:
            topic_name = item.properties.get("final_topic", "Unknown")
            if topic_name:
                topic_counts[topic_name] += 1

        filtered_topics = [
            (name, count) for name, count in topic_counts.items()
            if count >= min_posts
        ]
        filtered_topics.sort(
            key=lambda x: x[0] if sort_by == "name" else x[1],
            reverse=(sort_by == "count"),
        )

        total_posts = len(all_posts.objects)
        topic_list = [
            {
                "topic_name": name,
                "post_count": count,
                "percentage": round((count / total_posts) * 100, 2)
                if total_posts > 0 else 0,
            }
            for name, count in filtered_topics
        ]
        return {
            "success": True,
            "total_topics": len(topic_list),
            "topics": topic_list,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 10: RECENT POSTS
# =============================

def get_recent_posts(
    days: Optional[int] = None, 
    limit: int = 20, 
    topic_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Get the most recently published posts."""
    
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        
        # Build filters
        combined_filter = None
        
        if days is not None:
            threshold_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            threshold_iso = threshold_date + "T00:00:00Z"
            combined_filter = Filter.by_property("post_date").greater_or_equal(threshold_iso)
        
        if topic_filter:
            topic_filter_obj = Filter.by_property("final_topic").equal(topic_filter)
            if combined_filter is not None:
                combined_filter = combined_filter & topic_filter_obj
            else:
                combined_filter = topic_filter_obj
        
        # Build query kwargs
        query_kwargs = {
            "limit": limit,
            "return_properties": [
                "post_number", "post_title", "post_date", "final_topic",
                "topic_confidence", "post_content", "secondary_topics",
            ],
            "sort": Sort.by_property(name="post_number", ascending=False),
        }
        
        if combined_filter is not None:
            query_kwargs["filters"] = combined_filter
        
        results = post_collection.query.fetch_objects(**query_kwargs)
        formatted_results = []
        now = datetime.now()
        
        for obj in results.objects:
            props = obj.properties or {}
            post_date_iso = props.get("post_date")
            
            days_ago = None
            if post_date_iso:
                try:
                    post_date = datetime.fromisoformat(post_date_iso.replace("Z", "+00:00"))
                    days_ago = (now - post_date.replace(tzinfo=None)).days
                except Exception:
                    pass
            
            formatted_results.append({
                "post_number": props.get("post_number"),
                "title": props.get("post_title"),
                "date": post_date_iso,
                "primary_topic": props.get("final_topic"),
                "topic_confidence": props.get("topic_confidence"),
                "preview": (props.get("post_content") or "")[:150] + "...",
                "secondary_topics": props.get("secondary_topics"),
                "days_ago": days_ago,
            })
        
        period_str = "All time" if days is None else f"Last {days} days"
        
        return {
            "success": True,
            "period": period_str,
            "total_results": len(formatted_results),
            "topic_filter": topic_filter,
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()
# =============================
# TOOL 11: AGGREGATE POSTS
# =============================

def aggregate_posts(
    group_by: str = "topic", date_range: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Aggregate posts by topic, month, or year."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        filters = None
        
        if date_range:
            # FIX: Use simple string concatenation
            start_iso = date_range["start"] + "T00:00:00Z"
            end_iso = date_range["end"] + "T23:59:59Z"
            
            filters = (
                Filter.by_property("post_date").greater_or_equal(start_iso)
                & Filter.by_property("post_date").less_or_equal(end_iso)
            )

        results = post_collection.query.fetch_objects(
            filters=filters, limit=10000,
            return_properties=["final_topic", "post_date"]
        )

        aggregations = defaultdict(int)
        for obj in results.objects:
            key = "Unknown"
            if group_by == "topic":
                key = obj.properties.get("final_topic", "Unknown")
            elif group_by in ["month", "year"]:
                date_iso = obj.properties.get("post_date")
                if date_iso:
                    try:
                        dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                        if group_by == "month":
                            key = dt.strftime("%Y-%m")
                        else:
                            key = str(dt.year)
                    except:
                        key = "Unknown Date"
                else:
                    key = "Unknown Date"
            aggregations[key] += 1

        total = len(results.objects)
        formatted_aggs = [
            {
                "group": group,
                "count": count,
                "percentage": round((count / total) * 100, 2) if total > 0 else 0,
            }
            for group, count in sorted(
                aggregations.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return {
            "success": True,
            "grouped_by": group_by,
            "total_posts": total,
            "aggregations": formatted_aggs,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()

# =============================
# TOOL 12: SEARCH CHUNKS
# =============================

def search_chunks(
    query: str, limit: int = 10, post_title: Optional[str] = None
) -> Dict[str, Any]:
    """Search for specific passages within posts using semantic search."""
    client = get_weaviate_client()
    try:
        chunk_collection = client.collections.get("Chunk")

        try:
            query_vector = get_embedding_for_query(query)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate query vector: {e}",
                "query": query,
            }

        filters = None
        if post_title:
            filters = Filter.by_property("post_title").equal(post_title)

        results = chunk_collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=filters,
            return_properties=[
                "post_number", "post_title", "chunk_number",
                "chunk_text", "chunk_topic",
            ],
        )

        formatted_results = []
        for obj in results.objects:
            props = obj.properties
            formatted_results.append({
                "post_number": props.get("post_number"),
                "post_title": props.get("post_title"),
                "chunk_number": props.get("chunk_number"),
                "chunk_text": props.get("chunk_text")[:300] + "...",
                "topic": props.get("chunk_topic"),
            })

        return {
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}
    finally:
        if client:
            client.close()

# =============================
# TOOL 13: GET POSTS FOR DAILY
# =============================

def get_posts_for_daily(topic: str) -> Dict[str, Any]:
    """Fetch recent posts on a topic with current style guide."""
    try:
        posts_result = search_posts_by_topic(
            topic_name=topic,
            limit=10,
            fuzzy=True,
            include_secondary=False
        )
        
        posts = posts_result.get("results", []) if posts_result.get("success") else []
        posts_exist = len(posts) > 0
        
        style_guide = ""
        if os.path.exists(STYLE_GUIDE_PATH):
            with open(STYLE_GUIDE_PATH, "r") as f:
                style_guide = f.read()
        else:
            style_guide = (
                "Style guide not found. Create it with Sanjay Sahay's writing style:\n"
                "- ALL CAPS titles as bold statements\n"
                "- Short philosophical paragraphs\n"
                "- Title repeated for emphasis\n"
                "- Concise, impactful language\n"
                "- Sign with author name\n"
                "- 100-150 words typically"
            )
        
        return {
            "success": True,
            "topic": topic,
            "posts_count": len(posts),
            "posts": posts,
            "posts_exist": posts_exist,
            "style_guide": style_guide,
            "min_posts_for_pattern": MIN_POSTS_FOR_PATTERN,
            "message": (
                f"Fetched {len(posts)} existing posts on '{topic}'. "
                "Use these to check for redundancy. Generate new post using style guide."
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "topic": topic}

# =============================
# TOOL 14: ADD WRITING PATTERN
# =============================

def add_writing_pattern(pattern_description: str) -> Dict[str, Any]:
    """Add new writing pattern to style guide if not already present."""
    try:
        current_guide = ""
        if os.path.exists(STYLE_GUIDE_PATH):
            with open(STYLE_GUIDE_PATH, "r") as f:
                current_guide = f.read()
        
        if pattern_description.lower() in current_guide.lower():
            return {
                "success": False,
                "message": "Pattern already exists in style guide or very similar",
            }
        
        if "## DISCOVERED PATTERNS" in current_guide:
            updated_guide = current_guide.replace(
                "## DISCOVERED PATTERNS",
                f"## DISCOVERED PATTERNS\n- {pattern_description}",
            )
        else:
            updated_guide = (
                current_guide + f"\n\n## DISCOVERED PATTERNS\n- {pattern_description}\n"
            )
        
        with open(STYLE_GUIDE_PATH, "w") as f:
            f.write(updated_guide)
        
        return {
            "success": True,
            "message": f"✓ Pattern added: {pattern_description}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# =============================
# TOOL 15: GET STYLE GUIDE
# =============================

def get_style_guide() -> Dict[str, Any]:
    """Get current style guide content."""
    try:
        if os.path.exists(STYLE_GUIDE_PATH):
            with open(STYLE_GUIDE_PATH, "r") as f:
                style_guide = f.read()
            return {"success": True, "style_guide": style_guide}
        else:
            return {
                "success": False,
                "message": f"Style guide not found at {STYLE_GUIDE_PATH}",
                "style_guide": "",
            }
    except Exception as e:
        return {"success": False, "error": str(e), "chunk_number": props.get("chunk_number"),
                "chunk_text": props.get("chunk_text"),
                "topic": props.get("chunk_topic"),
            }

if __name__ == "__main__":
    # Test the tools
    print("🧪 Testing MCP Tools\n")
    
    def run_test(tool_name, tool_function, *args, **kwargs):
        print(f"Test: {tool_name}")
        try:
            result = tool_function(*args, **kwargs)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"--- TEST FAILED: {e} ---")
        print("\n" + "="*80 + "\n")    
    
