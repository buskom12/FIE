"""
GitHub интеграция для Hermes Bot
Управление репозиторием через GitHub API
"""

import os
import requests
from typing import Optional, List, Dict
from datetime import datetime


class GitHubManager:
    """Менеджер для работы с GitHub API"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.owner = "buskom12"
        self.repo = "FIE"
        self.base_url = f"https://api.github.com/repos/{self.owner}/{self.repo}"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    def get_repo_info(self) -> Dict:
        """Получить информацию о репозитории"""
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "name": data.get("name"),
                "description": data.get("description"),
                "stars": data.get("stargazers_count"),
                "forks": data.get("forks_count"),
                "open_issues": data.get("open_issues_count"),
                "default_branch": data.get("default_branch"),
                "updated_at": data.get("updated_at"),
                "size": data.get("size"),  # KB
                "language": data.get("language"),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_latest_commits(self, n: int = 5) -> List[Dict]:
        """Получить последние коммиты"""
        try:
            url = f"{self.base_url}/commits"
            params = {"per_page": n}
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )
            response.raise_for_status()
            commits = response.json()

            result = []
            for commit in commits:
                result.append({
                    "sha": commit["sha"][:7],
                    "message": commit["commit"]["message"].split("\n")[0],
                    "author": commit["commit"]["author"]["name"],
                    "date": commit["commit"]["author"]["date"],
                })
            return result

        except Exception as e:
            return [{"error": str(e)}]

    def get_branches(self) -> List[str]:
        """Получить список веток"""
        try:
            url = f"{self.base_url}/branches"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            branches = response.json()
            return [b["name"] for b in branches]

        except Exception as e:
            return [f"error: {str(e)}"]

    def create_issue(self, title: str, body: str, labels: Optional[List[str]] = None) -> Dict:
        """Создать issue"""
        try:
            url = f"{self.base_url}/issues"
            data = {
                "title": title,
                "body": body,
            }
            if labels:
                data["labels"] = labels

            response = requests.post(
                url, headers=self.headers, json=data, timeout=10
            )
            response.raise_for_status()
            issue = response.json()

            return {
                "number": issue["number"],
                "url": issue["html_url"],
                "state": issue["state"],
            }

        except Exception as e:
            return {"error": str(e)}

    def get_open_issues(self, limit: int = 5) -> List[Dict]:
        """Получить открытые issues"""
        try:
            url = f"{self.base_url}/issues"
            params = {"state": "open", "per_page": limit}
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )
            response.raise_for_status()
            issues = response.json()

            result = []
            for issue in issues:
                # Пропускаем pull requests
                if "pull_request" in issue:
                    continue

                result.append({
                    "number": issue["number"],
                    "title": issue["title"],
                    "state": issue["state"],
                    "created_at": issue["created_at"],
                    "url": issue["html_url"],
                })
            return result

        except Exception as e:
            return [{"error": str(e)}]

    def get_pull_requests(self, state: str = "open", limit: int = 5) -> List[Dict]:
        """Получить pull requests"""
        try:
            url = f"{self.base_url}/pulls"
            params = {"state": state, "per_page": limit}
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )
            response.raise_for_status()
            prs = response.json()

            result = []
            for pr in prs:
                result.append({
                    "number": pr["number"],
                    "title": pr["title"],
                    "state": pr["state"],
                    "created_at": pr["created_at"],
                    "url": pr["html_url"],
                    "head": pr["head"]["ref"],
                    "base": pr["base"]["ref"],
                })
            return result

        except Exception as e:
            return [{"error": str(e)}]

    def get_commit_activity(self) -> Dict:
        """Получить статистику активности коммитов"""
        try:
            url = f"{self.base_url}/stats/commit_activity"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                return {"error": "No commit activity data available"}

            # Последняя неделя
            last_week = data[-1]
            return {
                "week": datetime.fromtimestamp(last_week["week"]).strftime("%Y-%m-%d"),
                "total": last_week["total"],
                "days": last_week["days"],
            }

        except Exception as e:
            return {"error": str(e)}

    def format_repo_status(self) -> str:
        """Форматированный статус репозитория для Telegram"""
        info = self.get_repo_info()
        
        if "error" in info:
            return f"❌ Ошибка: {info['error']}"

        commits = self.get_latest_commits(3)
        
        text = f"""
🗂 *GitHub Репозиторий*

📦 Название: `{info['name']}`
⭐️ Stars: {info['stars']}
🍴 Forks: {info['forks']}
🐛 Open Issues: {info['open_issues']}
🌿 Default Branch: `{info['default_branch']}`
💾 Размер: {info['size']} KB
🔤 Язык: {info['language']}

📝 *Последние коммиты:*
"""
        
        for commit in commits[:3]:
            if "error" not in commit:
                date = datetime.fromisoformat(commit["date"].replace("Z", "+00:00"))
                text += f"\n`{commit['sha']}` - {commit['message'][:50]}"
                text += f"\n└ {commit['author']} • {date.strftime('%d.%m %H:%M')}\n"

        return text.strip()

    def format_issues(self) -> str:
        """Форматированный список issues"""
        issues = self.get_open_issues()

        if not issues:
            return "✅ Нет открытых issues"

        if "error" in issues[0]:
            return f"❌ Ошибка: {issues[0]['error']}"

        text = "🐛 *Открытые Issues:*\n\n"
        for issue in issues:
            created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
            text += f"#{issue['number']} - {issue['title'][:50]}\n"
            text += f"└ {created.strftime('%d.%m.%Y')}\n\n"

        return text.strip()

    def format_pull_requests(self) -> str:
        """Форматированный список PR"""
        prs = self.get_pull_requests()

        if not prs:
            return "✅ Нет открытых Pull Requests"

        if "error" in prs[0]:
            return f"❌ Ошибка: {prs[0]['error']}"

        text = "🔀 *Pull Requests:*\n\n"
        for pr in prs:
            created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
            text += f"#{pr['number']} - {pr['title'][:50]}\n"
            text += f"└ {pr['head']} → {pr['base']} • {created.strftime('%d.%m.%Y')}\n\n"

        return text.strip()
