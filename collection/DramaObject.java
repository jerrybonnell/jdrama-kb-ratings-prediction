import java.util.*;
import java.io.*;

/**
 * Class for storing information about a drama
 * as it appears in the artv website
 */

public class DramaObject {
  /* *************************************************
   * The class has many instance variables
   * ************************************************* */

  String season;
  String aname;
  String title;
  String station;
  String slot;
  String day;
  String startTime;
  String endTime;
  int episodes;
  int episodesRev;
  String link;
  ArrayList<String> actor;
  double average;
  ArrayList<Double> percentage;
  ArrayList<String> date;
  ArrayList<String> producer;
  ArrayList<String> chiefProducer;
  ArrayList<String> director;
  ArrayList<String> script;
  ArrayList<String> original;
  ArrayList<String> themeSong;
  ArrayList<String> sungInDrama;
  ArrayList<String> inDramaSong;
  ArrayList<String> endingMusic;
  ArrayList<String> openingMusic;
  ArrayList<String> themeMusic;
  ArrayList<String> imageMusic;
  ArrayList<String> titleMusic;

  /**
   * This constructor initializes the instance variables
   * that are lists
   */
  DramaObject() {
    actor = new ArrayList<String>();
    percentage = new ArrayList<Double>();
    date = new ArrayList<String>();
    producer = new ArrayList<String>();
    chiefProducer = new ArrayList<String>();
    director = new ArrayList<String>();
    script = new ArrayList<String>();
    original = new ArrayList<String>();
    themeSong = new ArrayList<String>();
    sungInDrama = new ArrayList<String>();
    inDramaSong = new ArrayList<String>();
    endingMusic = new ArrayList<String>();
    openingMusic = new ArrayList<String>();
    themeMusic = new ArrayList<String>();
    imageMusic = new ArrayList<String>();
    titleMusic = new ArrayList<String>();
  }

  /**
   * This constructor initializes the instance variables
   * that are lists. The only difference from the no-param
   * constructor is that this one assigns the season value.
   *
   * @param	season	a string specifying the seach info
   */
  DramaObject( String season ) {
    this();
    setSeason( season );
  }

  /* ********************************************************
   * Setters for assigning values to non-list instance variables
   * ******************************************************** */
  void setSeason( String s ) { season = s; }
  void setAName( String t ) { aname = t; }
  void setTitle( String t ) { title = t; }
  void setStation( String s ) { station = s; }
  void setSlot( String t ) { slot = t; }
  void setDay( String t ) { day = t; }
  void setStartTime( String t ) { startTime = t; }
  void setEndTime( String t ) { endTime = t; }
  void setEpisodes( int n ) { episodes = n; episodesRev = n; }
  void setEpisodes( int n, int m ) { episodes = n; episodesRev = m; }
  void setLink( String l ) { link = l; }
  void setAverage( double d ) { average = d; }

  /* ********************************************************
   * Methods for adding elements to lists
   * ******************************************************** */
  void addActor( String a ) { actor.add( a ); }
  void addPercentage( double r ) { percentage.add( r ); }
  void addPercentage( String w ) {
    if ( w.endsWith( "%" ) ) w = w.substring( 0, w.length() - 1 );
    percentage.add( Double.parseDouble( w ) );
  }
  void addDate( String d ) { date.add( d ); }
  /**
   * the elements after element-0 to the list
   */
  void addCDR( ArrayList<String> list, String[] w ) {
    for ( int j = 1; j < w.length; j ++ ) list.add( w[j] );
  }
  /**
   * the string array as an input and add informatiuon
   * @param	w	the string array
   *		the first element of w is the type
   *		the remainders are the entries
   */
  void addTeam( String[] w ) {
    switch ( w[0] ) {
      case "Ｐ": addCDR( producer, w ); break;
      case "チーフＰ": addCDR( chiefProducer, w ); break;
      case "演出": addCDR( director, w ); break;
      case "脚本": addCDR( script, w ); break;
      case "原作": addCDR( original, w ); break;
      case "主題歌": addCDR( themeSong, w ); break;
      case "挿入歌": addCDR( sungInDrama, w ); break;
      case "劇中歌": addCDR( inDramaSong, w ); break;
      case "ED曲": addCDR( endingMusic, w ); break;
      case "OP曲": addCDR( openingMusic, w ); break;
      case "テーマ曲": addCDR( themeMusic, w ); break;
      case "イメージ曲": addCDR( imageMusic, w ); break;
      case "タイトル曲": addCDR( titleMusic, w ); break;
      default: System.out.println( "addTeam. Not found. " + w[0] );
    }
  }


  /**
   * analyze lines
   * @param	input	List of string
   */
  void analyze( List<String> input ) {
    ////// local variables
    String[] parts;
    String w, u, v;
    int p, q, e1, e2, k1, k2;

    /* ****************************************************
     * preprocessing...
     * build the variable data from the input
     * where each tab and "etc" is removed and <br> is treated
     * as a line break
     * Empty lines are ignored
     * **************************************************** */
    ArrayList<String> data = new ArrayList<String>();
    for ( String x : input ) {
      x = x.replace( "\t", "" ).trim();
      parts = x.split( "<br>" );
      for ( int i = 0; i < parts.length; i ++ ) {
        p = parts[ i ].indexOf( " etc" );
        if ( p >= 0 ) parts[ i ] = parts[ i ].substring( 0, p );
        if ( parts[ i ].length() > 0 ) data.add( parts[ i ] );
      }
    }
    /* *****************************************************************
     * Find two viewer positions.
     * k1 is the position of the first episode
     * the first episode has information about the actors,
     * production team, etc
     * k2 is the position of the second episode
     * ***************************************************************** */
    k1 = -1;
    k2 = -1;
    for ( int i = 0; i < data.size(); i ++ ) {
      if ( k2 > 0 ) continue;
      if ( k1 >= 0 ) {
        if ( data.get( i ).startsWith( "<tr><td>2</td><td>" ) ) k2 = i;
      }
      else if ( data.get( i ).startsWith( "<tr><td>1</td><td>" ) ) k1 = i;
    }
    /* *****************************************************************
     * Flow of the analysis is as follows:
     *
     * aname
     * <table border="1" class="tableborder ar" align="center">
     * title, station
     * slot, link
     * 6 lines skipped
     * 1st, date, percentage
     * <td rowspan="10" class="left pl5px">
     * cast, etc., in multiple lines
     * </td>
     * </tr>
     * <tr><td>NUMBER</td><td>DATE</td><td ..>PERCENTAGE</td></tr>
     * ...
     * <tr><td>NUMBER</td><td>DATE</td><td ..>PERCENTAGE</td></tr>
     * <tr><th ..>平均視聴率：PERCENTAGE</td></tr>
     * </table>
    /* ***************************************************************** */

    /* *****************************************************************
     * aname
     * this is the element 0 of the data:
     * this is a local reference name (only valid in the file)
     * the same reference name can appear in other files
     * ***************************************************************** */
    w = data.get( 0 );
    p = w.indexOf( "\"" );
    q = w.indexOf( "\"", p + 1 );
    w = w.substring( p + 1, q );
    setAName( w );

      /* *************************************************************
       * title and station
       * this is the element 2 of the data:
       * the title and the station appear in between the 2nd and 3rd tags
       * the station name appears between the 2"　＜" and "＞" in the tag
       * ************************************************************* */
    w = data.get( 2 );
      /* *************************************************************
       * p = position of the 2nd ">"
       * q = position of the 3rd "<"
       * update w with the string between the two positions,
       * the update operation should be always successful
       * ************************************************************* */
    p = repPos( w, ">", 2 );
    q = repPos( w, "<", 3 );
    w = w.substring( p + 1, q );
      /* *************************************************************
       * p = position of "　＜"
       * if the pattern appears, split into two parts and store them
       * otherwise, use the string as the title and leave the station empty
       * ************************************************************* */
    p = w.indexOf( "　＜" ); // the space is in Japanese
    if ( p > 0 ) {
      u = w.substring( 0, p );
      v = w.substring( p + 2, w.length() - 1 );
      setTitle( u );
      setStation( v );
    }
    else {
      setTitle( w );
      setStation( "" );
    }

    /* *************************************************************
     * the slot information and the number(s) of episodes
     * this is element 3
     * ************************************************************* */
    w = data.get( 3 );
      /* *************************************************************
       * p = position of the 2nd ">"
       * q = position of the 3rd "<"
       * update w as the string between the two positions
       * save the string as the slot information
       * ************************************************************* */
    p = repPos( w, ">", 2 );
    q = repPos( w, "<", 3 );
    w = w.substring( p + 1, q );
    setSlot( w );
      /* *************************************************************
       * the first two characters are the day of week
       * then appears the time and the number(s) of episodes
       * set u to the time information, between the day of week and episodes
       * replace the japanese : with the english :
       * the first five characters are the start time
       * then after a dash, the next five are the ending time
       * ************************************************************* */
    setDay( w.substring( 0, 2 ) );
    p = w.indexOf( "　（" );
    u = w.substring( 2, p ).replace( "：", ":" );
    setStartTime( u.substring( 0, 5 ) );
    setEndTime( u.substring( 6 ) );
      /* *************************************************************
       * the numbers should appear between "　（全" and "回）"
       * if the pattern matches,
       * pass the string to setEpisodes to set the numbers;
       * the first number is the number appearing at the start
       * the second number is the number appearing at the end
       * If there is no change in the number, there will be only one number
       *   so the two numbers obtained are the same
       * otherwise, set -1 for both
       * ************************************************************* */
    p = w.indexOf( "　（全" );
    q = w.indexOf( "回）" );
    if ( p >= 0 && q > p ) {
      v = w.substring( p + 3, q );
      setEpisodes( forwardInt( v ), backwardInt( v ) );
    }
    else {
      setEpisodes( -1, -1 );
    }

    /* *************************************************************
     * the link String is element 3 and appears in the third tag
     * between two double quotation marks
     * ************************************************************* */
    w = data.get( 3 );
    p = repPos( w, "<", 3 );
    q = repPos( w, ">", 3 );
    w = w.substring( p + 1, q ); // update w
    p = repPos( w, "\"", 1 );
    q = repPos( w, "\"", 2 );
    w = w.substring( p + 1, q );
    setLink( w );

    /* *************************************************************
     * The avarage appears in the second to last element of data
     * between the Japanese colon and the english %
     * Parse it, if there are markers, and use 0 otherwise
     * ************************************************************* */
    w = data.get( data.size() - 2 );
    p = w.indexOf( "：" );
    q = w.indexOf( "%" );
    setAverage( ( p >= 0 && q >= 0 ) ?
        Double.parseDouble( w.substring( p + 1, q ) ) : 0.0 );

    /* *************************************************************
     * Add the first episode's %, which appears at position k1
     * ************************************************************* */
    addDateAndPercentage( data.get( k1 ) );

    /* *************************************************************
     * Add the personnel and team information appearing
     * between positions k1 and k2-1
     * Each element, if not startinb with "<", carries some inormation
     * Use isRoleString() to test if the string is a role giving string
     * Extracting the role information is by calling splitRole()
     * Otherwise, split the line with "、" as the delimiter and store
     * all the elements in the actor list.
     * ************************************************************* */
    for ( int j = k1; j < k2; j++ ) {
      w = data.get( j );
      if ( isRoleString( w ) ) {
        parts = splitRole( w );
        if ( parts != null ) {
          addTeam( parts );
        }
      }
      else if ( !w.startsWith( "<" ) ) {
        parts = w.split( "、" );
        for ( int k = 0; k < parts.length; k ++ ) {
          if ( parts[ k ].length() > 0 )
            addActor( parts[ k ] );
        }
      }
    }

    /* *************************************************************
     * Add the remaining episodes' %, which appears starting at k2
     * The end position is the size of the data - 3.
     * After adding the % values, if the final number of episodes
     * is -1, store the data's size in the final number.
     * ************************************************************* */
    for ( int j = k2; j < data.size() - 2; j++ ) {
      addDateAndPercentage( data.get( j ) );
    }
    if ( episodesRev == -1 && percentage.size() > 0 ) {
      episodesRev = percentage.size();
    }
  }

  /**
   * add the date and percentage to their lists
   * @param	w	input string
   *		the date is between the fourth and the fifth tag;
   *		the percentage should be found from the last %
   *		appearing in w
   */
  void addDateAndPercentage( String w ) {
    String u, v;
    int p, q;
    p = repPos( w, ">", 4 );
    q = repPos( w, "<", 5 );
    u = w.substring( p + 1, q );
    addDate( ( u.length() > 0 ) ? w.substring( p + 1, q ) : "N/A" );
    u = backPercentSearch( w );
    addPercentage( ( u.length() > 1 ) ?
        Double.parseDouble( u.substring( 0, u.length() - 1 ) ) : 0.0 );
  }

  /**
   * the role matching table
   * the name appearing in the html and the name used alternate
   */
  private static String[] ROLE_MATCHING = {
    "season", "season",
    "aname", "aname",
    "title", "title",
    "link", "link",
    "slot", "slot",
    "day", "day",
    "startTime", "startTime",
    "endTime", "endTime",
    "station", "station",
    "episodes", "episodes",
    "episodesRev", "episodesRev",
    "percentage", "percentage",
    "date", "date",
    "ave", "ave",
    "actor", "actor",
    "Ｐ", "producer",
    "チーフＰ", "chiefProducer",
    "演出", "director",
    "脚本", "script",
    "原作", "original",
    "主題歌", "themeSong",
    "劇中歌", "sungInDrama",
    "挿入歌", "inDramaSong",
    "ED曲", "endingMusic",
    "OP曲", "openingMusic",
    "テーマ曲", "themeMusic",
    "イメージ曲", "imageMusic",
    "タイトル曲", "titleMusic"
  };

  /**
   * @return	English day of week
   * @param	Japanese day of week
   */
  public static String dayToEnglish( String w ) {
    switch( w ) {
      case "日曜" : return "Sun";
      case "月曜" : return "Mon";
      case "火曜" : return "Tue";
      case "水曜" : return "Wed";
      case "木曜" : return "Thu";
      case "金曜" : return "Fri";
      case "土曜" : return "Sat";
      default: return "N/A";
    }
  }

  /**
   * @return	the heder string (for csv)
   */
  public static String header() {
    StringBuilder builder = new StringBuilder();
    builder.append( "\"" + ROLE_MATCHING[ 1 ] + "\"" );
    for ( int i = 3; i < ROLE_MATCHING.length; i += 2 ) {
      builder.append( ",\"" + ROLE_MATCHING[ i ] + "\"" );
    }
    return builder.toString();
  }

  /**
   * @return	encoding of the object (for csv)
   */
  public String encode() {
    return String.format(
      "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\","
    + "\"%d\",\"%d\",\"%s\",\"%s\",\"%.2f\",\"%s\","
    + "\"%s\",\"%s\",\"%s\",\"%s\","
    + "\"%s\",\"%s\",\"%s\",\"%s\","
    + "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"",
    season, aname, title, link, slot, dayToEnglish( day ),
        startTime, endTime, station,
    episodes, episodesRev, assembleD( percentage ), assemble( date ),
        average, assemble( actor ), 
    assemble( producer ), assemble( chiefProducer ),
        assemble( director ), assemble( script ),
    assemble( original ), assemble( themeSong ),
        assemble( sungInDrama ), assemble( inDramaSong ),
    assemble( endingMusic ), assemble( openingMusic ),
        assemble( themeMusic ), assemble( imageMusic ),
            assemble( titleMusic ) );
  }

  /**
   * @param	list	list of String
   * @return	a String listing of the elements
   */
  private String assemble( ArrayList<String> list ) {
    if ( list.size() == 0 ) return "";
    StringBuilder builder = new StringBuilder( list.get(0) );
    for ( int j = 1; j < list.size(); j ++ ) {
      builder.append( "," + list.get(j) );
    }
    return builder.toString();
  }

  /**
   * @param	list	list of Double
   * @return	a String listing of the numbers
   */
  private String assembleD( ArrayList<Double> list ) {
    if ( list.size() == 0 ) return "";
    StringBuilder builder = new StringBuilder();
    builder.append( String.format( "%.2f", list.get(0) ) );
    for ( int j = 1; j < list.size(); j ++ ) {
      builder.append( "," + String.format( "%.2f", list.get(j) ) );
    }
    return builder.toString();
  }

  /** print all the information */
  public void print() {
    System.out.printf( "----------%n" );
    System.out.printf( "season=%s%n", season );
    System.out.printf( "aname=%s%n", aname );
    System.out.printf( "title=%s%n", title );
    System.out.printf( "station=%s%n", station );
    System.out.printf( "slot=%s%n", slot );
    System.out.printf( "day=%s%n", day );
    System.out.printf( "startTime=%s%n", startTime );
    System.out.printf( "endTime=%s%n", endTime );
    System.out.printf( "episodes=%s%n", "" + episodes );
    System.out.printf( "episodesRev=%s%n", "" + episodesRev );
    System.out.printf( "link=%s%n", link );
    System.out.printf( "ave=%.2f%n", average );
    System.out.printf( "actor=" );
    for ( int j = 0; j < actor.size(); j ++ ) {
      System.out.printf( "%s", actor.get( j ) );
      System.out.printf( ( j < actor.size() - 1 ) ? "," : "%n" );
    }
    System.out.printf( "percentage=" );
    for ( int j = 0; j < percentage.size(); j ++ ) {
      System.out.printf( "%s", percentage.get( j ) );
      System.out.printf( ( j < percentage.size() - 1 ) ? "," : "%n" );
    }
    System.out.printf( "date=" );
    for ( int j = 0; j < date.size(); j ++ ) {
      System.out.printf( "%s", date.get( j ) );
      System.out.printf( ( j < date.size() - 1 ) ? "," : "%n" );
    }
  }

  /**
   * Location rep many occurrences of a pattern
   * @param	w	the input
   * @param	pat	the pattern
   * @param	rep	the repetitions
   * @return	if the number of occurrences < rep, -1;
   *		o.w., the position
   */
  public static int repPos( String w, String pat, int rep ) {
    if ( pat.length() == 0 ) return -1;
    int k = -1, r;
    for ( int j = 1; j <= rep; j ++ ) {
      if ( ( r = w.indexOf( pat, k + 1 ) ) < 0 ) return -1;
      k = r;
    }
    return k;
  }

  /**
   * Extract % String
   * @param	w	the input
   * @return	the last substring with %
   */
  public static String backPercentSearch( String w ) {
    int p = w.lastIndexOf( "%" );
    if ( p < 0 ) return "";
    int r = p - 1;
    char c;
    while ( r >= 0 && ( ( c = w.charAt( r ) )== '.' ||
        ( c >= '0' && c <= '9' ) ) ) r--;
    return w.substring( r + 1, p + 1 );
  }

  /**
   * @param	w	the input
   * @return	the first integer appearing in w
   */
  public static int forwardInt( String w ) {
    w = w.trim();
    int p = 0;
    char c;
    while ( p < w.length() &&
      ( c = w.charAt( p ) ) >= '0' && c <= '9' ) p ++;
    return Integer.parseInt( w.substring( 0, p ) );
  }
  /**
   * @param	w	the input
   * @return	the last integer appearing in w
   */
  public static int backwardInt( String w ) {
    w = w.trim();
    int p = w.length() - 1;
    char c;
    while ( p >= 0 &&
      ( c = w.charAt( p ) ) >= '0' && c <= '9' ) p --;
    return Integer.parseInt( w.substring( p + 1 ) );
  }

  /**
   * @param	w	the input
   * @return	if the String descrives a role
   */
  private static boolean isRoleString( String w ) {
    return w.indexOf( " ： " ) >= 0;
  }

  /**
   * @param	w	the input
   * @return	decomposed role string
   */
  private static String[] splitRole( String w ) {
    if ( !isRoleString( w ) ) return null;
    int p;
    w = w.replace( " etc", "" );
    String[] a, b, c;
    a = w.split( " ： " );
    if ( a.length == 1 ) return null;
    if ( ( p = a[1].indexOf( "『" ) ) > 0 ) {
      c = new String[ 3 ];
      c[0] = a[0];
      c[1] = a[1].substring( 0, p );
      c[2] = a[1].substring( p );
    }
    else {
      b = a[1].split( "、" );
      c = new String[ 1 + b.length ];
      c[0] = a[0];
      for ( int j = 0; j < b.length; j ++ )
        c[ j+1 ] = b[ j ];
    }
    return c;
  }

  private static final String TEST1 =
      "<tr><td>1</td><td>10/8</td><td class=\"center max\">" +
      "(69分)11.5%</td>";

  private static final String TEST2 =
      "演出 ： 三宅喜重、小松隆志、植田尚 etc";

  public static void testOne() {
    System.out.println( repPos( TEST1, "<", 4 ) );
    System.out.println( backPercentSearch( TEST1 ) );
    System.out.println( isRoleString( TEST2 ) );
    for ( String x : splitRole( TEST2 ) ) System.out.println( " " + x );
  }

  public static void main( String[] args ) throws FileNotFoundException {
    System.out.println( DramaObject.header() );
  }
}


///////////////////////////////////////////////////////////////
//////// Below is a sample section from a viewership webpage
///////////////////////////////////////////////////////////////
/*
<a name="kekkondekinaiotoko"></a>
<table border="1" class="tableborder ar" align="center">
	<tr><th colspan="4" class="red">まだ結婚できない男　＜フジテレビ＞</th></tr>
	<tr><th colspan="4">火曜22：00〜22：54　（全10回）　<a href="https://www.ktv.jp/kekkondekinaiotoko/" target="_blank">⇒公式サイト</a></th></tr>
	<tr>
		<th width="8%" align="center">回</th>
		<th width="11%" align="center">日</th>
		<th width="20%" align="center">視聴率</th>
		<th width="61%" align="center">キャスト＆スタッフ</th>
	</tr>
	<tr><td>1</td><td>10/8</td><td class="center max">(69分)11.5%</td>
		<td rowspan="10" class="left pl5px">
阿部寛、吉田羊、<br>深川麻衣、塚本高史、咲妃みゆ、平祐奈、<br>阿南敦子、奈緒、荒井敦史、小野寺ずる、<br>美音、RED RICE、デビット伊東、不破万作、<br>三浦理恵子、尾美としのり、稲森いずみ、草笛光子 etc<br><br>脚本 ： 尾崎将也<br>
チーフＰ ： 安藤和久、東城祐司<br>
Ｐ ： 米田孝、伊藤達哉、木曽貴美子<br>
演出 ： 三宅喜重、小松隆志、植田尚 etc
<br><br>
主題歌 ： 持田香織『まだスイミー』<br>
		</td>
	</tr>
	<tr><td>2</td><td>10/15</td><td class="center min">7.7%</td></tr>
	<tr><td>3</td><td>10/22</td><td class="center">10.0%</td></tr>
	<tr><td>4</td><td>10/29</td><td class="center">9.5%</td></tr>
	<tr><td>5</td><td>11/5</td><td class="center">10.0%</td></tr>
	<tr><td>6</td><td>11/12</td><td class="center">8.9%</td></tr>
	<tr><td>7</td><td>11/19</td><td class="center">8.5%</td></tr>
	<tr><td>8</td><td>11/26</td><td class="center">8.6%</td></tr>
	<tr><td>9</td><td>12/3</td><td class="center">9.0%</td></tr>
	<tr><td>10</td><td>12/10</td><td class="center">9.7%</td></tr>
	<tr><th colspan="4" class="center bold">平均視聴率：9.40%</td></tr>
</table>
 */
/*
actor
aname
ave
date
day
endTime
episodes
episodesRev
link
percentage
season
slot
startTime
station
title
Ｐ:producer
チーフＰ:chief_producer
原作:original
演出:director
脚本:script
主題歌:main_song
劇中歌:sung_in_drama_song
挿入歌:inside_song
ED曲:ending_music
OP曲:opening_music
テーマ曲:theme_music
イメージ曲:image_music
タイトル曲:title_music
*/
