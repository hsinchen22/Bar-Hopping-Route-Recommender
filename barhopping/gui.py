import gradio as gr
import asyncio
from typing import List, Dict, Any, Tuple
from barhopping.retriever.vector_search import get_vector_search
from barhopping.path_finder import PathFinder
from barhopping.config import BARS_DB
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
from barhopping.logger import logger

class BarHoppingGUI:
    def __init__(self):
        """Initialize the GUI with vector search and path finding capabilities."""
        self.vector_search = get_vector_search()
        self.path_finder = PathFinder(BARS_DB)
        self.browser = None
        
    def _init_browser(self):
        """Initialize the browser if not already initialized."""
        if self.browser is None:
            try:
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")  # Run in headless mode
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                self.browser = webdriver.Chrome(options=options)
                logger.info("Browser initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize browser: {str(e)}")
                raise
        
    def _cleanup_browser(self):
        """Clean up browser resources."""
        if self.browser is not None:
            try:
                self.browser.quit()
                self.browser = None
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
        
    def bar_html(self, name: str, url: str, address: str, img: str, summary: str) -> str:
        """Generate HTML for a bar card."""
        # Handle missing, protocol-relative, or invalid image URLs
        img_url = None
        if img:
            if img.startswith(("http://", "https://")):
                img_url = img
            elif img.startswith("//"):
                img_url = "https:" + img
        if img_url:
            img_html = f'<img src="{img_url}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 12px; margin: 10px 0;" />'
        else:
            img_html = '<div style="width: 100%; height: 200px; background-color: #2a2a2a; border-radius: 12px; margin: 10px 0; display: flex; align-items: center; justify-content: center;"><p style="color: #666; font-size: 14px;">No image available</p></div>'

        return f"""
        <div style="padding: 10px; font-family: 'Segoe UI', sans-serif; background-color: #1e1e1e; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.2); margin-bottom: 20px;">
            <p style="font-size: 20px; font-weight: bold; color: white;">{name}
                <a href="{url}" target="_blank" style="margin-left: 8px; font-size: 14px; text-decoration: none; color: #fbbf24">
                    Google Maps »
                </a>
            </p>
            <p style="font-size: 14px; color: #bbb; text-align: left; margin-top: -4px">📍{address}</p>
            {img_html}
            <p style="font-size: 14px; color: #bbb; text-align: left;">{summary}</p>
        </div>
        """
        
    def map_html(self, url: str) -> str:
        """Generate HTML for the route map."""
        return f"""
        <div style="padding: 10px; font-family: 'Segoe UI', sans-serif; background-color: #1e1e1e; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.2); margin-bottom: 20px;">
            <p style="font-size: 20px; font-weight: bold; color: white;">🗺️ Night Crawl Route
                <a href="{url}" target="_blank" style="margin-left: 8px; font-size: 14px; text-decoration: none; color: #fbbf24">
                    View Route on Google Maps »
                </a>
            </p>
        </div>
        """
        
    def path_html(self, distance: str) -> str:
        """Generate HTML for the path section between bars."""
        # Convert distance to English format if it's in meters
        if "公尺" in distance:
            meters = distance.replace("公尺", "").strip()
            distance = f"{meters} meters"
        elif 'm' in distance.lower() and 'km' not in distance.lower():
            meters = distance.lower().replace('m', '').strip()
            distance = f"{meters} meters"
        elif '公里' in distance:
            km = distance.replace('公里', '').strip()
            distance = f"{km} kilometers"
        elif 'km' in distance.lower():
            km = distance.lower().replace('km', '').strip()
            distance = f"{km} kilometers"
            
        return f"""
        <div style="padding: 8px; font-family: 'Segoe UI', sans-serif; background-color: #2a2a2a; border-radius: 12px; margin: 10px 0; text-align: center;">
            <p style="font-size: 14px; color: #fbbf24; margin: 0;">🚶‍♂️ Walking Distance: {distance}</p>
        </div>
        """
        
    async def get_recommendation(self, query: str) -> List[Dict[str, Any]]:
        """Get bar recommendations based on the query."""
        results = self.vector_search.search(query, top_k=5)
        return results
        
    async def make_routes(self, bar_ids: List[int]) -> Tuple[str, List[str]]:
        """Generate route for the selected bars using Google Maps."""
        try:
            # Initialize browser if needed
            self._init_browser()
            
            # Get addresses for the bars
            addresses = self.path_finder.get_bar_addresses(bar_ids)
            
            # Find optimal path using PathFinder
            distances = {}
            for i in range(len(bar_ids)):
                for j in range(i + 1, len(bar_ids)):
                    distances[(bar_ids[i], bar_ids[j])] = 1
                    distances[(bar_ids[j], bar_ids[i])] = 1
            
            # Find optimal path
            _, optimal_path = self.path_finder.find_optimal_path(bar_ids, distances)
            
            # Reorder addresses based on optimal path
            optimal_addresses = [addresses[bar_ids.index(id)] for id in optimal_path]
            
            # Open Google Maps
            url = "https://www.google.com/maps/dir/"
            self.browser.get(url)
            self.browser.maximize_window()

            # Click walking mode - wait for the button to be clickable
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "m6Uuef"))
            )
            travel_btns = self.browser.find_elements(By.CLASS_NAME, "m6Uuef")
            for btn in travel_btns:
                if btn.get_attribute("data-tooltip") == "Walking":
                    btn.click()
                    break

            # Add first two addresses
            inputs = self.browser.find_elements(By.CLASS_NAME, "tactile-searchbox-input")
            inputs[0].send_keys(optimal_addresses[0])
            inputs[1].send_keys(optimal_addresses[1])
            inputs[1].send_keys(Keys.ENTER)
            await asyncio.sleep(2)

            # Get distances between consecutive bars
            path_distances = []
            for i in range(len(optimal_addresses) - 1):
                # Add remaining addresses
                if i > 0:
                    add_btn = self.browser.find_elements(By.CLASS_NAME, "fC7rrc")[-1]
                    add_btn.click()
                    await asyncio.sleep(1)
                    input = self.browser.find_elements(By.CLASS_NAME, "tactile-searchbox-input")[-1]
                    input.send_keys(optimal_addresses[i + 1])
                    input.send_keys(Keys.ENTER)
                    await asyncio.sleep(4)

                # Get the distance for this segment
                try:
                    distance_element = WebDriverWait(self.browser, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "Fk3sm"))
                    )
                    distance = distance_element.text
                    path_distances.append(distance)
                except Exception as e:
                    logger.error(f"Error getting distance: {str(e)}")
                    path_distances.append("Distance unavailable")

            return self.browser.current_url, path_distances
            
        except WebDriverException as e:
            logger.error(f"Browser error: {str(e)}")
            # Try to reinitialize browser
            self._cleanup_browser()
            self._init_browser()
            raise
        except Exception as e:
            logger.error(f"Error generating route: {str(e)}")
            raise
        
    async def bar_recommendation(self, message: str, history: List[Tuple[str, str]]):
        """Handle the chat interface for bar recommendations."""
        try:
            response = []
            bars = await self.get_recommendation(message)
            
            # Get actual bar IDs from the database results
            bar_ids = [bar["id"] for bar in bars]
            
            # Start route generation task
            route_task = asyncio.create_task(self.make_routes(bar_ids))
            
            # Display bars with path information
            for i, bar in enumerate(bars):
                html = self.bar_html(
                    bar["name"],
                    bar.get("URL", "#"),
                    bar["address"],
                    bar.get("photo", ""),
                    bar.get("summary", "No description available")
                )
                response.append(html)
                
                # Add path information after each bar except the last one
                if i < len(bars) - 1:
                    # Wait for route task to complete if we need the distances
                    if i == 0:
                        url, path_distances = await route_task
                    
                    # Add path distance
                    path_html = self.path_html(path_distances[i])
                    response.append(path_html)
                
                yield response
            
            # Add route at the bottom
            url, _ = await route_task
            route_html = self.map_html(url)
            response.append(route_html)
            yield response
            
        except Exception as e:
            logger.error(f"Error in bar recommendation: {str(e)}")
            yield ["Sorry, an error occurred while processing your request."]
        
    def launch(self):
        """Launch the Gradio interface."""
        try:
            with gr.Blocks(fill_height=True, css="""
            .chatbox {
                flex: 1;
                height: 100%;
                overflow-y: auto !important;
                padding: 20px;
            }
            .avatar-container {
                width: 50px !important;
                height: 50px !important;
                border-radius: 50% !important;
            }
            .gradio-container {
                height: 100vh;
                background-color: #0e0e0e !important;
            }
            .contain {
                overflow-y: auto !important;
                max-height: 100vh !important;
            }
            footer {
                display: none !important;
            }
            .message {
                max-width: 100% !important;
            }
            .message img {
                max-width: 100% !important;
                height: auto !important;
            }
            .description {
                color: white !important;
            }
            """) as demo:
                gr.ChatInterface(
                    fn=self.bar_recommendation,
                    description="<strong><span style='color:#fbbf24;'>RunTini</span></strong> <span style='color:white;'>Bar Hopping Route Recommender</span>",
                    textbox=gr.Textbox(
                        placeholder="Think aesthetics, music, drinks, and crowd...",
                        submit_btn=True
                    ),
                    chatbot=gr.Chatbot(
                        elem_classes=["chatbox"],
                        placeholder="Let's map out your perfect night — pick a vibe or tell me yours! 🍸✨",
                        bubble_full_width=False,
                        avatar_images=["./images/user_avatar.png", "./images/bot_avatar.png"],
                        show_label=False,
                        type="messages"
                    ),
                    examples=[
                        "Cozy bars with dim lighting and jazz music for a relaxed evening",
                        "Trendy rooftop bars with great views and photogenic cocktails",
                        "Bars with retro arcade vibes and playful, neon-lit interiors",
                        "Speakeasy-style spots with hidden entrances and vintage aesthetics"
                    ],
                    type="messages"
                )
                
            demo.launch(share=True, debug=True)
            
        finally:
            # Clean up browser resources
            self._cleanup_browser()