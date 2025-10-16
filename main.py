#!/usr/bin/env python3
"""
Main entry point for the Advanced Document Search Engine

Usage:
    python -m search_engine.main --web         # Start web interface
    python -m search_engine.main --search     # Command line search
    python -m search_engine.main --index      # Index documents
    python -m search_engine.main --stats      # Show statistics
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .integration import initialize_search_engine
from .web_app import run_web_app
from .config import search_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def command_line_search():
    """Interactive command-line search interface"""
    try:
        search_engine, _, api = initialize_search_engine()

        print("🔍 Advanced Document Search Engine")
        print("=" * 50)

        while True:
            try:
                query = input("\nEnter search query (or 'quit' to exit): ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if not query:
                    continue

                print(f"\nSearching for: '{query}'")

                # Perform search
                result = api.search(query, limit=10)

                if not result['success']:
                    print(f"❌ Search failed: {result['error']}")
                    continue

                data = result['data']

                if not data['results']:
                    print("📭 No results found.")
                    continue

                print(f"\n✅ Found {data['total_results']} results in {data['search_time']:.3f}s:")
                print("-" * 50)

                for i, result_item in enumerate(data['results'], 1):
                    doc = result_item['document']
                    print(f"{i}. {doc.get('title', doc.get('file_name', 'Untitled'))}")
                    print(f"   Type: {doc.get('content_type', 'Unknown')} | Size: {doc.get('file_size_human', 'Unknown')}")
                    print(f"   Path: {doc.get('file_path', 'Unknown')}")
                    if result_item.get('snippet'):
                        print(f"   Preview: {result_item['snippet'][:100]}...")
                    print(f"   Score: {result_item['score']:.3f}")
                    print()

                # Show facets if available
                if data.get('facets'):
                    print("📊 Search Facets:")
                    for facet_type, facet_data in data['facets'].items():
                        if facet_data:
                            print(f"  {facet_type.title()}:")
                            for value, count in list(facet_data.items())[:5]:
                                print(f"    - {value}: {count}")
                            print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.exception("Command line search error")

    except Exception as e:
        print(f"❌ Failed to initialize search engine: {str(e)}")
        logger.exception("Initialization error")
        return 1

    return 0


def index_documents(base_path: str = None):
    """Index documents from file system"""
    try:
        search_engine, _, api = initialize_search_engine()

        print("📚 Document Indexing")
        print("=" * 30)

        if base_path:
            print(f"Indexing documents from: {base_path}")
            indexed_count = search_engine.reindex_all_documents(base_path)
        else:
            print("Indexing documents from existing catalog...")
            indexed_count = search_engine.reindex_all_documents()

        print(f"✅ Successfully indexed {indexed_count} documents")

        # Show statistics
        stats = api.get_stats()
        if stats['success']:
            print("\n📊 Index Statistics:")
            print(f"  Total documents: {stats['data']['total_documents']}")
            print(f"  Vector store chunks: {stats['data']['vector_store_stats']['total_chunks']}")
            print(f"  Popular documents: {stats['data']['popular_documents_count']}")

        return 0

    except Exception as e:
        print(f"❌ Indexing failed: {str(e)}")
        logger.exception("Indexing error")
        return 1


def show_statistics():
    """Show search engine statistics"""
    try:
        search_engine, _, api = initialize_search_engine()

        print("📊 Search Engine Statistics")
        print("=" * 35)

        stats = api.get_stats()
        if not stats['success']:
            print(f"❌ Failed to get stats: {stats['error']}")
            return 1

        data = stats['data']

        print(f"Total Documents: {data['total_documents']}")
        print(f"Vector Store Chunks: {data['vector_store_stats']['total_chunks']}")
        print(f"Vector Dimensions: {data['vector_store_stats']['dimensions']}")
        print(f"Embedding Model: {data['vector_store_stats']['model_name']}")
        print(f"Total Searches: {data['total_searches']}")
        print(f"Popular Documents Count: {data['popular_documents_count']}")

        # Show popular documents
        popular = api.get_popular_documents(5)
        if popular['success'] and popular['data']:
            print("\n🔥 Most Popular Documents:")
            for i, item in enumerate(popular['data'], 1):
                doc_id = item['doc_id']
                popularity = item['popularity']
                print(f"{i}. {doc_id}")
                print(f"   Searches: {popularity['total_searches']} | Clicks: {popularity['total_clicks']}")
                print(f"   Popularity Score: {popularity['popularity_score']:.2f}")
                print()

        return 0

    except Exception as e:
        print(f"❌ Failed to get statistics: {str(e)}")
        logger.exception("Statistics error")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Document Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m search_engine.main --web                    # Start web interface
  python -m search_engine.main --search                 # Interactive CLI search
  python -m search_engine.main --index                  # Index all documents
  python -m search_engine.main --index /path/to/docs    # Index specific directory
  python -m search_engine.main --stats                  # Show statistics
        """
    )

    parser.add_argument(
        '--web',
        action='store_true',
        help='Start web interface'
    )

    parser.add_argument(
        '--search',
        action='store_true',
        help='Interactive command-line search'
    )

    parser.add_argument(
        '--index',
        nargs='?',
        const='',
        metavar='PATH',
        help='Index documents (from existing catalog or specified path)'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show search engine statistics'
    )

    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Web server host (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Web server port (default: 5000)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle mutually exclusive operations
    operations = [args.web, args.search, args.index, args.stats]
    if sum(operations) > 1:
        print("❌ Please specify only one operation at a time")
        return 1

    if args.index is not None:
        # Index documents
        base_path = args.index if args.index else None
        return index_documents(base_path)

    elif args.web:
        # Start web interface
        print("🚀 Starting web interface...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Debug: {args.debug}")
        print(f"   URL: http://{args.host}:{args.port}")
        print()

        try:
            run_web_app(host=args.host, port=args.port, debug=args.debug)
            return 0
        except KeyboardInterrupt:
            print("\n👋 Web server stopped")
            return 0
        except Exception as e:
            print(f"❌ Web server failed: {str(e)}")
            logger.exception("Web server error")
            return 1

    elif args.search:
        # Interactive search
        return command_line_search()

    elif args.stats:
        # Show statistics
        return show_statistics()

    else:
        # Default to web interface
        parser.print_help()
        print("\nStarting web interface by default...")
        try:
            run_web_app(debug=args.debug)
            return 0
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
        except Exception as e:
            print(f"❌ Failed to start web interface: {str(e)}")
            return 1


if __name__ == '__main__':
    sys.exit(main())