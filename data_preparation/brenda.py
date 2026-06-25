import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import time
import re
from tqdm import tqdm

class BrendaScraper:
    def __init__(self, username, password):
        self.base_url = "https://www.brenda-enzymes.org"
        self.session = requests.Session()
        self.login(username, password)

    def login(self, username, password):
        """Log in to the BRENDA website."""
        login_url = f"{self.base_url}/login.php"
        self.session.get(login_url)

        login_data = {
            'username': username,
            'password': password,
            'submit': 'Login'
        }
        response = self.session.post(login_url, data=login_data)

        if "Login failed" in response.text:
            raise Exception("Login failed: check your username and password.")

        print("Login successful.")

    def get_ec_numbers(self):
        """Fetch all EC numbers from BRENDA."""
        url = f"{self.base_url}/all_enzymes.php"
        ec_numbers = []

        try:
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                print(f"Error: server returned {response.status_code}.")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')

            for link in soup.find_all('a'):
                href = link.get('href', '')
                if 'enzyme.php?ecno=' in href:
                    ec = href.split('ecno=')[1]
                    if ec.replace('.', '').isdigit():
                        ec_numbers.append(ec)

            print(f"Found {len(ec_numbers)} EC numbers.")
            return list(set(ec_numbers))

        except requests.exceptions.Timeout:
            print("Error: request timed out.")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error: request failed - {str(e)}")
            return []
        except Exception as e:
            print(f"Error: unexpected error while fetching EC numbers - {str(e)}")
            return []

    def get_enzyme_data(self, ec_number):
        """Fetch detailed data for a given EC number."""
        url = f"{self.base_url}/enzyme.php?ecno={ec_number}"

        try:
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                print(f"Error: server returned {response.status_code} for EC {ec_number}.")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            enzyme_data = []

            # Parse enzyme name from page title
            enzyme_name = '-'
            title = soup.find('title')
            if title:
                title_text = title.text
                if 'Information on EC' in title_text:
                    enzyme_name = title_text.split(' - ')[1].strip()

            # 1. Find the Turnover Number table
            turnover_table = None
            for div in soup.find_all('div', {'id': lambda x: x and x.startswith('tab')}):
                header_row = div.find('div', {'class': 'row'})
                if not header_row:
                    continue

                headers = header_row.find_all('div', {'class': 'header'})
                header_texts = [h.text.strip() for h in headers]
                required_headers = [
                    "TURNOVER NUMBER [1/s]",
                    "SUBSTRATE",
                    "ORGANISM",
                    "UNIPROT",
                    "COMMENTARY",
                    "LITERATURE",
                    "IMAGE"
                ]

                if all(header in header_texts for header in required_headers):
                    turnover_table = div
                    break

            if turnover_table:
                rows = turnover_table.find_all('div', {'class': ['row rgrey2', 'row rgrey1', 'hidden rgrey2', 'hidden rgrey1']})

                for row in rows:
                    if 'entries' in row.text:
                        continue

                    cells = row.find_all('div', {'class': 'cell'})
                    if len(cells) >= 7:
                        entry = {
                            'enzyme_name': enzyme_name,
                            'ec_number': ec_number,
                            'turnover_number': '-',
                            'substrate': '-',
                            'organism': '-',
                            'uniprot': '-',
                            'commentary': '-'
                        }

                        # Extract turnover number
                        turnover_cell = cells[0]
                        turnover_span = turnover_cell.find('span')
                        turnover_text = turnover_span.text.strip() if turnover_span else turnover_cell.text.strip()

                        turnover_values = re.findall(r'[\d.]+', turnover_text)
                        if turnover_values:
                            try:
                                values = [float(v) for v in turnover_values]
                                entry['turnover_number'] = sum(values) / len(values)
                            except ValueError:
                                entry['turnover_number'] = '-'

                        # Extract substrate
                        substrate_cell = cells[1]
                        substrate_span = substrate_cell.find('span')
                        substrate_text = substrate_span.text.strip() if substrate_span else substrate_cell.text.strip()
                        if substrate_text and substrate_text != '-':
                            entry['substrate'] = substrate_text

                        # Extract organism
                        organism_cell = cells[2]
                        organism_span = organism_cell.find('span')
                        if organism_span:
                            organism_link = organism_span.find('a')
                            organism_text = organism_link.text.strip() if organism_link else organism_span.text.strip()
                        else:
                            organism_link = organism_cell.find('a')
                            organism_text = organism_link.text.strip() if organism_link else organism_cell.text.strip()

                        if organism_text and organism_text != '-' and 'entries' not in organism_text:
                            organism_text = re.sub(r'<[^>]+>', '', organism_text)
                            organism_text = re.sub(r'&[^;]+;', '', organism_text)
                            organism_text = re.sub(r'\s+', ' ', organism_text).strip()
                            entry['organism'] = organism_text

                        # Extract UniProt ID
                        uniprot_cell = cells[3]
                        uniprot_span = uniprot_cell.find('span')
                        if uniprot_span:
                            uniprot_link = uniprot_span.find('a')
                            uniprot_text = uniprot_link.text.strip() if uniprot_link else uniprot_span.text.strip()
                        else:
                            uniprot_link = uniprot_cell.find('a')
                            uniprot_text = uniprot_link.text.strip() if uniprot_link else uniprot_cell.text.strip()

                        if not uniprot_text or uniprot_text == '-':
                            continue

                        entry['uniprot'] = uniprot_text

                        # Extract commentary
                        commentary_cell = cells[4]
                        commentary_span = commentary_cell.find('span')
                        commentary_text = commentary_span.text.strip() if commentary_span else commentary_cell.text.strip()
                        if commentary_text and commentary_text != '-':
                            entry['commentary'] = commentary_text

                        if 'entries' not in entry['organism']:
                            enzyme_data.append(entry)

            # 2. Find the Protein Variants table
            variants_table = soup.find('div', {'id': lambda x: x and x.startswith('tab') and 'PROTEIN VARIANTS' in soup.find('div', {'id': x}).text})
            if variants_table:
                rows = variants_table.find_all('div', {'class': ['row rgrey2', 'row rgrey1', 'hidden rgrey2', 'hidden rgrey1']})

                for row in rows:
                    cells = row.find_all('div', {'class': 'cell'})
                    if len(cells) >= 3:
                        variant = cells[0].find('span')
                        if variant:
                            variant_name = variant.text.strip()

                            # Extract organism
                            organism_cell = cells[1]
                            organism_link = organism_cell.find('a')
                            variant_organism = organism_link.text.strip() if organism_link else '-'

                            # Extract UniProt ID
                            uniprot_cell = cells[2]
                            uniprot = uniprot_cell.text.strip()

                            if uniprot == '-':
                                continue

                            search_url = f"{self.base_url}/search_result.php?a=8&W[2]={variant_name}&W[1]={ec_number}&T[1]=1&V[3]=1&V[5]=1&os=1"

                            try:
                                response = self.session.get(search_url, timeout=30)

                                if response.status_code != 200:
                                    print(f"Error: server returned {response.status_code} for variant {variant_name}.")
                                    continue

                                soup = BeautifulSoup(response.text, 'html.parser')

                                reference_links = soup.find_all('a', href=lambda x: x and 'literature.php' in x)

                                if not reference_links:
                                    continue

                                for ref_link in reference_links:
                                    ref_url = f"{self.base_url}/{ref_link['href']}"
                                    ref_response = self.session.get(ref_url, timeout=30)
                                    ref_soup = BeautifulSoup(ref_response.text, 'html.parser')

                                    # Find turnover number table in the reference page
                                    turnover_table = None
                                    for section in ref_soup.find_all('section', {'class': 'literature-table-section'}):
                                        h2 = section.find('h2')
                                        if h2 and h2.text.strip() == 'Turnover Number [1/s]':
                                            turnover_table = section
                                            break

                                    if turnover_table:
                                        rows = turnover_table.find('tbody').find_all('tr')

                                        for row in rows:
                                            cells = row.find_all('td')
                                            if len(cells) >= 5:
                                                literature_organism = cells[4].text.strip()
                                                if literature_organism:
                                                    literature_organism = re.sub(r'<[^>]+>', '', literature_organism)
                                                    literature_organism = re.sub(r'&[^;]+;', '', literature_organism)
                                                    literature_organism = re.sub(r'\s+', ' ', literature_organism).strip()

                                                if literature_organism != variant_organism:
                                                    continue

                                                entry = {
                                                    'enzyme_name': enzyme_name,
                                                    'ec_number': ec_number,
                                                    'turnover_number': '-',
                                                    'substrate': '-',
                                                    'organism': variant_organism,
                                                    'uniprot': uniprot,
                                                    'commentary': '-'
                                                }

                                                # Extract turnover number from min/max range
                                                turnover_min = cells[0].text.strip()
                                                turnover_max = cells[1].text.strip()

                                                try:
                                                    if turnover_min not in ('-', ''):
                                                        min_val = float(turnover_min)
                                                        if turnover_max not in ('-', ''):
                                                            max_val = float(turnover_max)
                                                            entry['turnover_number'] = (min_val + max_val) / 2
                                                        else:
                                                            entry['turnover_number'] = min_val
                                                    elif turnover_max not in ('-', ''):
                                                        entry['turnover_number'] = float(turnover_max)
                                                except ValueError:
                                                    entry['turnover_number'] = '-'

                                                substrate = cells[2].text.strip()
                                                if substrate:
                                                    entry['substrate'] = substrate

                                                commentary = cells[3].text.strip()
                                                if commentary and commentary != '-':
                                                    entry['commentary'] = commentary

                                                enzyme_data.append(entry)

                            except requests.exceptions.Timeout:
                                print(f"Error: request timed out for variant {variant_name}.")
                                continue
                            except requests.exceptions.RequestException as e:
                                print(f"Error: request failed for variant {variant_name} - {str(e)}")
                                continue
                            except Exception as e:
                                print(f"Error: unexpected error for variant {variant_name} - {str(e)}")
                                continue

            print(f"EC {ec_number}: collected {len(enzyme_data)} data points.")
            return enzyme_data

        except requests.exceptions.Timeout:
            print(f"Error: request timed out for EC {ec_number}.")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error: request failed for EC {ec_number} - {str(e)}")
            return []
        except Exception as e:
            print(f"Error: unexpected error for EC {ec_number} - {str(e)}")
            return []

    def save_to_json(self, data, filename):
        """Save data to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved JSON: {filename}")

    def save_to_csv(self, data, filename):
        """Save data to a CSV file."""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Saved CSV: {filename}")

    def collect_all_data(self):
        """Collect data for all EC numbers and save incrementally."""
        ec_numbers = self.get_ec_numbers()

        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)
        json_path = os.path.join(data_dir, 'brenda_data.json')

        all_data = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} existing entries from {json_path}")

        # Resume from a specific EC number if needed
        target_ec = "2.3.1.219"
        start_index = ec_numbers.index(target_ec) if target_ec in ec_numbers else 0
        ec_numbers = ec_numbers[start_index:]

        batch_size = 10
        for i in range(0, len(ec_numbers), batch_size):
            batch = ec_numbers[i:i+batch_size]
            print(f"\n[{i+1}-{min(i+batch_size, len(ec_numbers))}/{len(ec_numbers)}] Processing EC numbers...")

            batch_data = []
            for ec in batch:
                try:
                    data = self.get_enzyme_data(ec)
                    if data:
                        batch_data.extend(data)
                    time.sleep(5)
                except Exception as e:
                    print(f"Error processing EC {ec}: {str(e)}")
                    time.sleep(10)

            all_data.extend(batch_data)

            # Save after each batch
            with open(json_path, 'w') as f:
                json.dump(all_data, f, indent=4)

            csv_path = os.path.join(data_dir, 'brenda_data.csv')
            df = pd.DataFrame(all_data)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(all_data)} total entries → {json_path}, {csv_path}")

            time.sleep(30)

        print(f"\nData collection complete. Total: {len(all_data)} entries.")

    def edit_file(self, target_file, instructions, code_edit):
        """Overwrite a file with new content."""
        with open(target_file, 'w') as f:
            f.write(code_edit)

    def modify_json_structure(self):
        """Flatten the JSON structure so each (turnover, substrate, organism) is a separate entry."""
        json_path = os.path.join(os.getcwd(), 'data', 'brenda_data.json')

        with open(json_path, 'r') as f:
            data = json.load(f)

        new_data = []
        for entry in data:
            for i in range(len(entry['turnover_numbers'])):
                new_entry = {
                    'enzyme_name': entry['enzyme_name'],
                    'ec_number': entry['ec_number'],
                    'turnover_number': entry['turnover_numbers'][i],
                    'substrate': entry['substrates'][i] if i < len(entry['substrates']) else '',
                    'organism': entry['organisms'][i] if i < len(entry['organisms']) else '',
                    'commentary': entry['comments'][i] if i < len(entry['comments']) else ''
                }
                new_data.append(new_entry)

        with open(json_path, 'w') as f:
            json.dump(new_data, f, indent=4)

        print(f"JSON structure updated. Total entries: {len(new_data)}.")

def main():
    username = USERNAME
    password = PASSWORD

    scraper = BrendaScraper(username, password)

    try:
        scraper.collect_all_data()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
